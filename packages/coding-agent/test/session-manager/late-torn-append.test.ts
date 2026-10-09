import { appendFileSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { SessionManager } from "../../src/core/session-manager.ts";

// temporalio/pi-temporal#51: a superseded writer can leave a torn line after another manager
// opened the session, even after its write guard passes. Every later append must end that line
// without rewriting earlier bytes. A complete batch must reload with every entry, not be
// discarded as incomplete because its first line joined the stale fragment.
describe("SessionManager: a late torn append", () => {
	let dir: string;
	const cut = '{"type":"custom","customType":"stale","data":';

	beforeEach(() => {
		dir = mkdtempSync(join(tmpdir(), "pi-late-torn-"));
	});

	afterEach(() => {
		rmSync(dir, { recursive: true, force: true });
	});

	it.each(["single", "batch"])("keeps complete %s appends after repeated late cuts", (mode) => {
		const seed = SessionManager.create(dir, dir);
		seed.appendMessage({ role: "user", content: "seed", timestamp: 1 });
		const file = seed.getSessionFile()!;
		const current = SessionManager.open(file);

		// First after open, then after an earlier successful append. Neither cut belongs to
		// this manager, so it cannot rely on remembering its own failed writes.
		for (const n of [1, 2]) {
			const before = readFileSync(file, "utf8");
			let guards = 0;
			current.setWriteGuard(() => {
				// A batch checks each entry and checks again immediately before persistence.
				if (++guards === (mode === "batch" ? 3 : 1)) appendFileSync(file, cut);
			});
			if (mode === "batch") {
				current.batch(() => {
					current.appendCustomEntry(`current-${n}-a`, n);
					current.appendCustomEntry(`current-${n}-b`, n);
				});
			} else {
				current.appendCustomEntry(`current-${n}`, n);
			}
			expect.soft(guards).toBe(mode === "batch" ? 3 : 1);
			expect.soft(readFileSync(file, "utf8").startsWith(before + cut)).toBe(true);
			// Soft assertions keep the second cut exercised on the unfixed code too.
			expect.soft(SessionManager.open(file).getEntries()).toEqual(current.getEntries());
		}
	});

	it("opens a torn current-version session without writing, then checks the guard before appending", () => {
		const seed = SessionManager.create(dir, dir);
		seed.appendMessage({ role: "user", content: "seed", timestamp: 1 });
		const file = seed.getSessionFile()!;
		appendFileSync(file, cut);
		const before = readFileSync(file, "utf8");

		const current = SessionManager.open(file);
		expect(current.getEntries()).toEqual(seed.getEntries());
		expect.soft(readFileSync(file, "utf8")).toBe(before);
		current.setWriteGuard(() => {
			throw new Error("superseded");
		});
		expect(() => current.appendCustomEntry("refused", 1)).toThrow("superseded");
		expect.soft(readFileSync(file, "utf8")).toBe(before);

		let guards = 0;
		current.setWriteGuard(() => guards++);
		current.appendCustomEntry("kept", 2);
		expect(guards).toBe(1);
		expect(readFileSync(file, "utf8").startsWith(before)).toBe(true);
		expect(SessionManager.open(file).getEntries()).toEqual(current.getEntries());
	});
});
