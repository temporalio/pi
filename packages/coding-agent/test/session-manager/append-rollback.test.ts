import type * as fs from "node:fs";
import { existsSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SessionManager } from "../../src/core/session-manager.ts";

const state = vi.hoisted(() => ({
	failNextAppend: false,
	failNextWrite: false,
	tearNextAppend: false,
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
				throw new Error("disk said no");
			}
			if (state.tearNextAppend) {
				// Part of the line lands, then the disk fills up.
				state.tearNextAppend = false;
				actual.appendFileSync(args[0], String(args[1]).slice(0, 12));
				throw new Error("disk full");
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

	it("cuts off the part of a line a failed append left, so the retry is read back", () => {
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

	it("leaves the file alone when it lost the file before cutting a failed append back", () => {
		const manager = SessionManager.create(dir, join(dir, "sessions"));
		manager.appendMessage(user("one"));
		const file = manager.getSessionFile()!;
		// The guard lets the append through, then says no: another writer took the file while the
		// append was failing, and may have appended after it.
		let asked = 0;
		manager.setWriteGuard(() => {
			asked++;
			if (asked > 1) throw new Error("another writer has the session");
		});

		const before = readFileSync(file).length;
		state.tearNextAppend = true;
		expect(() => manager.appendMessage(user("two"))).toThrow("disk full");
		expect(asked).toBe(2);
		// Not cut back. What follows the torn part may be the new writer's.
		expect(readFileSync(file).length).toBe(before + 12);
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
