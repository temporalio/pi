import { existsSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SessionManager } from "../../src/core/session-manager.ts";

const state = vi.hoisted(() => ({ failNextAppend: false }));

vi.mock("node:fs", async (importOriginal) => {
	const actual = await importOriginal<typeof import("node:fs")>();
	return {
		...actual,
		appendFileSync: ((...args: Parameters<typeof actual.appendFileSync>) => {
			if (state.failNextAppend) {
				state.failNextAppend = false;
				throw new Error("disk said no");
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
});
