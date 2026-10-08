import type * as fs from "node:fs";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { loadEntriesFromFile, SessionManager } from "../../src/core/session-manager.ts";

const state = vi.hoisted(() => ({
	failNextAppend: false,
	failNextWrite: false,
	tearNextAppend: false,
	// Runs inside a failing append, before it throws, as a stalled write while a newer writer
	// takes over.
	duringFailedAppend: undefined as (() => void) | undefined,
	dieInNextAppend: false,
	appends: 0,
	writes: 0,
}));

vi.mock("node:fs", async (importOriginal) => {
	const actual = await importOriginal<typeof fs>();
	return {
		...actual,
		writeFileSync: ((...args: Parameters<typeof actual.writeFileSync>) => {
			state.writes++;
			if (state.failNextWrite) {
				state.failNextWrite = false;
				throw new Error("disk said no");
			}
			return actual.writeFileSync(...args);
		}) as typeof actual.writeFileSync,
		appendFileSync: ((...args: Parameters<typeof actual.appendFileSync>) => {
			state.appends++;
			if (state.failNextAppend) {
				state.failNextAppend = false;
				const during = state.duringFailedAppend;
				state.duringFailedAppend = undefined;
				during?.();
				throw new Error("disk said no");
			}
			if (state.tearNextAppend) {
				// Part of the line lands, then the disk fills up.
				state.tearNextAppend = false;
				actual.appendFileSync(args[0], String(args[1]).slice(0, 12));
				throw new Error("disk full");
			}
			if (state.dieInNextAppend) {
				// A short write: the first line lands, then the process is gone before the rest.
				state.dieInNextAppend = false;
				const text = String(args[1]);
				actual.appendFileSync(args[0], text.slice(0, text.indexOf("\n") + 1));
				throw new Error("process died");
			}
			return actual.appendFileSync(...args);
		}) as typeof actual.appendFileSync,
	};
});

describe("SessionManager: a failed append", () => {
	let dir: string;

	beforeEach(() => {
		const unique = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
		dir = join(tmpdir(), `pi-append-rollback-${unique}`);
		mkdirSync(dir, { recursive: true });
	});

	afterEach(() => {
		if (existsSync(dir)) rmSync(dir, { recursive: true });
	});

	const user = (text: string) => ({
		role: "user" as const,
		content: [{ type: "text" as const, text }],
		timestamp: Date.now(),
	});

	it("takes the entry back out of memory, so a retry writes it once", () => {
		const manager = SessionManager.create(dir, join(dir, "sessions"));
		const first = manager.appendMessage(user("one"));

		state.failNextAppend = true;
		expect(() => manager.appendMessage(user("two"))).toThrow("disk said no");

		// Memory rolled back with the failed write: the retry parents to the entry the file
		// holds, not to one that never reached it.
		const second = manager.appendMessage(user("two"));
		const entries = readFileSync(manager.getSessionFile()!, "utf8")
			.split("\n")
			.filter((line) => line.trim().length > 0)
			.map((line) => JSON.parse(line))
			.filter((entry) => entry.type === "message");
		expect(entries.map((entry) => entry.id)).toEqual([first, second]);
		expect(entries[1].parentId).toBe(first);
	});

	it("writes the first entries of a session in one write", () => {
		const manager = SessionManager.create(dir, join(dir, "sessions"));
		manager.appendCustomEntry("setup", { n: 1 });
		manager.appendCustomEntry("setup", { n: 2 });

		// Setup entries wait for the conversation, then go to the file together.
		state.writes = 0;
		manager.appendMessage(user("one"));
		expect(state.writes).toBe(1);
		expect(
			readFileSync(manager.getSessionFile()!, "utf8")
				.split("\n")
				.filter((l) => l.trim()),
		).toHaveLength(4);
	});

	it("removes a first write that failed part way, so the session can still be written", () => {
		const manager = SessionManager.create(dir, join(dir, "sessions"));

		state.failNextWrite = true;
		expect(() => manager.appendMessage(user("one"))).toThrow("disk said no");
		expect(existsSync(manager.getSessionFile()!)).toBe(false);

		const first = manager.appendMessage(user("one"));
		const ids = readFileSync(manager.getSessionFile()!, "utf8")
			.split("\n")
			.filter((line) => line.trim().length > 0)
			.map((line) => JSON.parse(line))
			.filter((entry) => entry.type === "message")
			.map((entry) => entry.id);
		expect(ids).toEqual([first]);
	});

	it("ends the part of a line a failed append left, so the retry is read back", () => {
		const manager = SessionManager.create(dir, join(dir, "sessions"));
		const first = manager.appendMessage(user("one"));

		state.tearNextAppend = true;
		expect(() => manager.appendMessage(user("two"))).toThrow("disk full");
		const second = manager.appendMessage(user("two"));

		const ids = SessionManager.open(manager.getSessionFile()!)
			.getBranch()
			.filter((entry) => entry.type === "message")
			.map((entry) => entry.id);
		expect(ids).toEqual([first, second]);
	});

	it("keeps what a newer writer appended while a stale append failed", () => {
		const manager = SessionManager.create(dir, join(dir, "sessions"));
		const first = manager.appendMessage(user("one"));
		const file = manager.getSessionFile()!;

		let newer: string | undefined;
		state.failNextAppend = true;
		state.duringFailedAppend = () => {
			newer = SessionManager.open(file).appendMessage(user("newer"));
		};
		expect(() => manager.appendMessage(user("stale"))).toThrow("disk said no");

		const ids = SessionManager.open(file)
			.getBranch()
			.filter((entry) => entry.type === "message")
			.map((entry) => entry.id);
		expect(ids).toEqual([first, newer]);
	});

	it("keeps a failed first write once the guard refuses, since another writer may own it", () => {
		const manager = SessionManager.create(dir, join(dir, "sessions"));
		let checks = 0;
		manager.setWriteGuard(() => {
			checks++;
			if (checks > 1) throw new Error("superseded");
		});

		state.failNextWrite = true;
		expect(() => manager.appendMessage(user("one"))).toThrow("disk said no");
		expect(existsSync(manager.getSessionFile()!)).toBe(true);
	});

	it("asks the write guard again when a batch commits", () => {
		const manager = SessionManager.create(dir, join(dir, "sessions"));
		const first = manager.appendMessage(user("one"));
		const lines = () =>
			readFileSync(manager.getSessionFile()!, "utf8")
				.split("\n")
				.filter((l) => l.trim());
		const before = lines().length;
		let asked = 0;
		manager.setWriteGuard(() => {
			// Owned while the batch is built, lost by the time it is written.
			if (++asked > 2) throw new Error("not ours to write");
		});

		expect(() =>
			manager.batch(() => {
				manager.appendCustomEntry("note", { n: 1 });
				manager.appendCustomEntry("note", { n: 2 });
			}),
		).toThrow("not ours to write");
		expect(manager.getLeafId()).toBe(first);
		expect(lines().length).toBe(before);
	});

	it("takes a label back too when its batch fails", () => {
		const manager = SessionManager.create(dir, join(dir, "sessions"));
		const first = manager.appendMessage(user("one"));

		state.failNextAppend = true;
		expect(() => manager.batch(() => manager.appendLabelChange(first, "kept?"))).toThrow("disk said no");
		expect(manager.getLabel(first)).toBeUndefined();
		expect(manager.getLeafId()).toBe(first);
	});

	it("writes a batch with one append, and takes all of it back when that append fails", () => {
		const manager = SessionManager.create(dir, join(dir, "sessions"));
		const first = manager.appendMessage(user("one"));
		const lines = () =>
			readFileSync(manager.getSessionFile()!, "utf8")
				.split("\n")
				.filter((l) => l.trim());
		const before = lines().length;

		state.failNextAppend = true;
		expect(() =>
			manager.batch(() => {
				manager.appendCustomEntry("note", { n: 1 });
				manager.appendCustomEntry("note", { n: 2 });
			}),
		).toThrow("disk said no");
		expect(manager.getLeafId()).toBe(first);
		expect(lines().length).toBe(before);

		state.appends = 0;
		manager.batch(() => {
			manager.appendCustomEntry("note", { n: 1 });
			manager.appendCustomEntry("note", { n: 2 });
		});
		expect(state.appends).toBe(1);
		expect(lines().length).toBe(before + 2);
	});
});

describe("SessionManager: a batch cut short on disk", () => {
	let dir: string;

	beforeEach(() => {
		const unique = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
		dir = join(tmpdir(), `pi-batch-frame-${unique}`);
		mkdirSync(dir, { recursive: true });
	});

	afterEach(() => {
		if (existsSync(dir)) rmSync(dir, { recursive: true });
	});

	const user = (text: string) => ({
		role: "user" as const,
		content: [{ type: "text" as const, text }],
		timestamp: Date.now(),
	});
	const fileLines = (file: string) =>
		readFileSync(file, "utf8")
			.split("\n")
			.filter((l) => l.trim())
			.map((l) => JSON.parse(l));
	const notes = (manager: SessionManager) =>
		manager
			.getEntries()
			.filter((e) => e.type === "custom")
			.map((e) => (e as { data?: { n: number } }).data?.n);
	const writeNotes = (manager: SessionManager) =>
		manager.batch(() => {
			manager.appendCustomEntry("note", { n: 1 });
			manager.appendCustomEntry("note", { n: 2 });
			manager.appendCustomEntry("note", { n: 3 });
		});

	it("loads a complete batch, without its frame", () => {
		const manager = SessionManager.create(dir, join(dir, "sessions"));
		manager.appendMessage(user("one"));
		writeNotes(manager);
		const file = manager.getSessionFile()!;

		const framed = fileLines(file).filter((e) => e.batch);
		expect(framed).toHaveLength(3);
		expect(new Set(framed.map((e) => e.batch.id)).size).toBe(1);
		expect(framed.every((e) => e.batch.size === 3)).toBe(true);

		const reopened = SessionManager.open(file);
		expect(notes(reopened)).toEqual([1, 2, 3]);
		expect(reopened.getEntries().some((e) => "batch" in e)).toBe(false);
		expect(reopened.getEntries()).toEqual(manager.getEntries());
	});

	it("frames a batch that is the session's first write", () => {
		const manager = SessionManager.create(dir, join(dir, "sessions"));
		manager.appendCustomEntry("setup", { n: 0 });
		manager.batch(() => {
			manager.appendMessage(user("one"));
			manager.appendCustomEntry("note", { n: 1 });
		});
		const lines = fileLines(manager.getSessionFile()!);
		expect(lines.map((e) => Boolean(e.batch))).toEqual([false, false, true, true]);
		expect(SessionManager.open(manager.getSessionFile()!).getEntries()).toEqual(manager.getEntries());
	});

	it("frames a batch held in memory until the session's first write", () => {
		const manager = SessionManager.create(dir, join(dir, "sessions"));
		manager.batch(() => {
			manager.appendModelChange("anthropic", "claude-opus-5-5");
			manager.appendThinkingLevelChange("high");
		});
		manager.appendCustomEntry("setup", { n: 0 });
		manager.appendMessage(user("one"));
		const file = manager.getSessionFile()!;
		const lines = fileLines(file);
		expect(lines.map((e) => Boolean(e.batch))).toEqual([false, true, true, false, false]);

		// Cut the file after the model change, as a power loss in the first write can.
		const raw = readFileSync(file, "utf8").split("\n");
		writeFileSync(file, `${raw.slice(0, 2).join("\n")}\n`);
		const reopened = SessionManager.open(file);
		expect(reopened.getEntries()).toEqual([]);
		expect(reopened.buildSessionContext().model).toBeNull();
	});

	it("leaves single appends and one-entry batches unframed", () => {
		const manager = SessionManager.create(dir, join(dir, "sessions"));
		manager.appendMessage(user("one"));
		manager.batch(() => manager.appendCustomEntry("note", { n: 1 }));
		manager.appendMessage(user("two"));
		expect(fileLines(manager.getSessionFile()!).some((e) => "batch" in e)).toBe(false);
	});

	it("drops a batch whose last lines never reached the file", () => {
		const manager = SessionManager.create(dir, join(dir, "sessions"));
		const first = manager.appendMessage(user("one"));
		writeNotes(manager);
		const file = manager.getSessionFile()!;
		const lines = readFileSync(file, "utf8")
			.split("\n")
			.filter((l) => l.trim());
		// Keep the first framed line whole and cut the second one part way.
		writeFileSync(file, `${lines.slice(0, -2).join("\n")}\n${lines[lines.length - 2].slice(0, 20)}`);

		const reopened = SessionManager.open(file);
		expect(notes(reopened)).toEqual([]);
		expect(reopened.getLeafId()).toBe(first);
	});

	it("loads a file without frames exactly as written", () => {
		const file = join(dir, "old.jsonl");
		const entries = [
			{ type: "session", version: 3, id: "s1", timestamp: "2026-01-01T00:00:00.000Z", cwd: dir },
			{
				type: "message",
				id: "a1",
				parentId: null,
				timestamp: "2026-01-01T00:00:01.000Z",
				message: { role: "user", content: [{ type: "text", text: "hi" }], timestamp: 1 },
			},
			{ type: "custom", id: "a2", parentId: "a1", timestamp: "2026-01-01T00:00:02.000Z", customType: "x", data: 1 },
		];
		writeFileSync(file, entries.map((e) => `${JSON.stringify(e)}\n`).join(""));
		expect(loadEntriesFromFile(file)).toEqual(entries);
		expect(SessionManager.open(file).getEntries()).toEqual(entries.slice(1));
	});

	it("does not commit a batch twice when the writer died part way and the turn was retried", () => {
		const manager = SessionManager.create(dir, join(dir, "sessions"));
		const first = manager.appendMessage(user("one"));
		const file = manager.getSessionFile()!;
		// The lease is gone by the time the append fails, so the dying writer cannot cut its
		// lines back and the file keeps the first line of the batch.
		let asked = 0;
		manager.setWriteGuard(() => {
			if (++asked > 4) throw new Error("another writer has the session");
		});
		state.dieInNextAppend = true;
		expect(() => writeNotes(manager)).toThrow("process died");
		expect(fileLines(file).filter((e) => e.type === "custom")).toHaveLength(1);

		// A new writer takes the session over and retries the boundary.
		const retry = SessionManager.open(file);
		expect(notes(retry)).toEqual([]);
		expect(retry.getLeafId()).toBe(first);
		writeNotes(retry);
		retry.appendMessage(user("two"));

		const reopened = SessionManager.open(file);
		expect(notes(reopened)).toEqual([1, 2, 3]);
		expect(reopened.getEntries()).toEqual(retry.getEntries());
	});
});
