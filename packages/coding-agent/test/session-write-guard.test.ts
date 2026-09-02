import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { SessionManager } from "../src/core/session-manager.ts";

// A driver that owns the session file through a lock has one place it can be sure it is still the
// owner: immediately before the write. Checking before a model call is not that place, because the
// write the model call ends with lands minutes later, and by then another attempt of the same
// activity can hold the lock. Two writers do not corrupt this file, they branch it, and the branch
// nobody reads is the half of a turn that disappears.
const msg = (content: string) => ({ role: "user", content, timestamp: Date.now() }) as const;

describe("session write guard", () => {
	const open = () => {
		const root = mkdtempSync(join(tmpdir(), "pi-write-guard-"));
		const sessionDir = join(root, "sessions");
		const sessions = SessionManager.create(root, sessionDir);
		// The manager's own view of the session, which is the tree a later reader walks. Counting
		// entries here rather than lines in the file keeps the assertion about what was appended
		// instead of about when the file happens to be flushed.
		return { sessions, entries: () => sessions.getEntries().length };
	};

	it("refuses a write once the guard says so, and lets the rest through", () => {
		const { sessions } = open();
		let entitled = true;
		sessions.setWriteGuard(() => {
			if (!entitled) throw new Error("lost the session lock");
		});

		sessions.appendMessage(msg("one"));
		const before = sessions.getEntries().length;
		expect(before).toBeGreaterThan(0);

		entitled = false;
		expect(() => sessions.appendMessage(msg("two"))).toThrow("lost the session lock");
		// Nothing was appended, so no second branch was started off the same leaf.
		expect(sessions.getEntries().length).toBe(before);

		entitled = true;
		sessions.appendMessage(msg("three"));
		expect(sessions.getEntries().length).toBe(before + 1);
	});

	// Every append goes through one funnel, so a guard on it covers the entry kinds a driver never
	// writes itself but an extension might.
	it("covers every entry kind, not only messages", () => {
		const { sessions } = open();
		sessions.appendMessage(msg("seed"));
		sessions.setWriteGuard(() => {
			throw new Error("lost the session lock");
		});
		expect(() => sessions.appendThinkingLevelChange("high")).toThrow("lost the session lock");
		expect(() => sessions.appendMessage(msg("no"))).toThrow("lost the session lock");
	});

	it("is off by default, so a session with no driver writes as it always did", () => {
		const { sessions } = open();
		expect(() => sessions.appendMessage(msg("one"))).not.toThrow();
		sessions.setWriteGuard(undefined);
		expect(() => sessions.appendMessage(msg("two"))).not.toThrow();
	});
});
