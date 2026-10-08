// A turn driven as a model call, its tool calls and a seal, against the session it writes to.
// The loop's own tests cover the parts; what is pinned here is the session file, which is what
// an outside driver actually keeps: the same turn split three ways has to leave the same
// transcript, and a call's result must not reach the file before the step is sealed.
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, type AgentMessage, type AgentTool, type TurnToolCallOutcome } from "@earendil-works/pi-agent-core";
import { getCurrentSystemPrompt } from "@earendil-works/pi-ai";
import {
	type AssistantMessage,
	type AssistantMessageEvent,
	type Context,
	EventStream,
	getModel,
} from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AgentSession } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { discoverAndLoadExtensions } from "../src/core/extensions/loader.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";
import { createTestResourceLoader } from "./utilities.ts";

/** What the inline test extensions report back, through globals they can reach. */
const testGlobals = globalThis as typeof globalThis & {
	reopenedTurnEnds?: number;
	reopenedContinued?: boolean;
	reopenedSettles?: number;
	steppedBeforeSettle?: number;
	steppedContinued?: boolean;
	steppedTurnEnds?: number;
};

class MockAssistantStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor() {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") return event.message;
				if (event.type === "error") return event.error;
				throw new Error("Unexpected event type");
			},
		);
	}
}

const usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function assistant(
	content: AssistantMessage["content"],
	stopReason: AssistantMessage["stopReason"] = "stop",
): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "anthropic-messages",
		provider: "anthropic",
		model: "mock",
		usage,
		stopReason,
		timestamp: Date.now(),
	};
}

/** Two calls, then an answer. Enough turn for a step boundary to be visible. */
const script = (): AssistantMessage[] => [
	assistant(
		[
			{ type: "toolCall", id: "call_1", name: "dummy", arguments: { q: "one" } },
			{ type: "toolCall", id: "call_2", name: "dummy", arguments: { q: "two" } },
		],
		"toolUse",
	),
	assistant([{ type: "text", text: "both done" }]),
];

const toolSchema = Type.Object({ q: Type.String() });

interface Harness {
	session: AgentSession;
	sessionManager: SessionManager;
	ran: string[];
	/** What each provider request carried. */
	requests: Context[];
	asked: () => number;
}

describe("stepped turn", () => {
	let tempDir: string;
	const built: AgentSession[] = [];

	beforeEach(() => {
		const unique = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
		tempDir = join(tmpdir(), `pi-stepped-turn-${unique}`);
		mkdirSync(tempDir, { recursive: true });
	});

	afterEach(() => {
		for (const session of built.splice(0)) {
			session.dispose();
		}
		if (tempDir && existsSync(tempDir)) rmSync(tempDir, { recursive: true });
	});

	async function createSession(
		name: string,
		options: {
			responses?: AssistantMessage[];
			reuse?: string;
			extension?: string;
			fastRetry?: boolean;
			/** The session's own custom system prompt. */
			systemPrompt?: string;
			/** The tool asks the turn to stop. */
			terminate?: boolean;
			/** The tool runs until its signal aborts. */
			hang?: boolean;
			/** The first provider request runs until its signal aborts. */
			hangModel?: boolean;
		} = {},
	): Promise<Harness> {
		const ran: string[] = [];
		const requests: Context[] = [];
		const responses = options.responses ?? script();
		let count = 0;
		const tool: AgentTool<typeof toolSchema, { q: string }> = {
			name: "dummy",
			label: "Dummy",
			description: "Records what it was asked",
			parameters: toolSchema,
			async execute(_id, params, signal) {
				ran.push(params.q);
				if (options.hang) {
					await new Promise((_resolve, reject) => {
						signal?.addEventListener("abort", () => reject(new Error(`stopped ${params.q}`)), { once: true });
					});
				}
				const result = { content: [{ type: "text" as const, text: `did ${params.q}` }], details: { q: params.q } };
				return options.terminate ? { ...result, terminate: true } : result;
			},
		};
		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model, systemPrompt: "Test", tools: [] },
			streamFn: (_model, context, streamOptions) => {
				requests.push(context);
				const stream = new MockAssistantStream();
				if (options.hangModel && requests.length === 1) {
					const aborted = assistant([], "aborted");
					stream.push({ type: "start", partial: aborted });
					streamOptions?.signal?.addEventListener(
						"abort",
						() => stream.push({ type: "error", reason: "aborted", error: aborted }),
						{ once: true },
					);
					return stream;
				}
				queueMicrotask(() => {
					const message = responses[Math.min(count, responses.length - 1)];
					count++;
					stream.push({ type: "start", partial: { ...message, content: [] } });
					stream.push({ type: "done", reason: "stop", message });
				});
				return stream;
			},
		});

		const sessionDir = join(tempDir, name);
		mkdirSync(sessionDir, { recursive: true });
		// Reusing a file is what the worker path does: every activity opens the same session.
		const sessionManager = options.reuse
			? SessionManager.open(options.reuse)
			: SessionManager.create(sessionDir, join(sessionDir, "sessions"));
		const settingsManager = SettingsManager.create(sessionDir, sessionDir);
		if (options.fastRetry) {
			settingsManager.applyOverrides({ retry: { enabled: true, maxRetries: 3, baseDelayMs: 1 } });
		}
		const authStorage = AuthStorage.create(join(sessionDir, "auth.json"));
		const modelRegistry = await createModelRegistry(authStorage, sessionDir);
		await authStorage.modify("anthropic", async () => ({ type: "api_key", key: "test-key" }));
		let extensionsResult: Awaited<ReturnType<typeof discoverAndLoadExtensions>> | undefined;
		if (options.extension) {
			const extensionsDir = join(sessionDir, ".pi", "extensions");
			mkdirSync(extensionsDir, { recursive: true });
			writeFileSync(join(extensionsDir, "e.ts"), options.extension);
			extensionsResult = await discoverAndLoadExtensions([], sessionDir, sessionDir);
		}

		const session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: sessionDir,
			modelRuntime: getModelRuntime(modelRegistry),
			resourceLoader: {
				...createTestResourceLoader(extensionsResult ? { extensionsResult } : undefined),
				getSystemPrompt: () => options.systemPrompt,
			},
			baseToolsOverride: { dummy: tool },
		});
		built.push(session);
		return { session, sessionManager, ran, requests, asked: () => count };
	}

	/** Messages as the session file holds them, so persistence is what is compared. */
	function persisted(sessionManager: SessionManager): AgentMessage[] {
		const file = sessionManager.getSessionFile();
		if (!file || !existsSync(file)) {
			return [];
		}
		return readFileSync(file, "utf8")
			.split("\n")
			.filter((line) => line.trim().length > 0)
			.map((line) => JSON.parse(line))
			.filter((entry) => entry.type === "message")
			.map((entry) => entry.message as AgentMessage);
	}

	/** Roles, call ids and text, with the parts that cannot match (timestamps) left out. */
	function shape(messages: AgentMessage[]) {
		return messages.map((message) => {
			if (message.role === "toolResult") {
				return { role: message.role, call: message.toolCallId, error: message.isError };
			}
			if (message.role === "assistant") {
				return { role: message.role, content: message.content.map((c) => c.type) };
			}
			return { role: message.role };
		});
	}

	/** Drive one turn as a model call, its calls, and a seal, the way a durable driver does. */
	async function driveStepped(harness: Harness): Promise<void> {
		for (;;) {
			const model = await harness.session.modelCall();
			const results: TurnToolCallOutcome[] = [];
			if (!model.ended) {
				for (const call of model.toolCalls) {
					const result = await harness.session.runToolCall(call.id);
					if (result) results.push(result);
				}
			}
			const { done } = await harness.session.sealStep(results);
			if (done) return;
		}
	}

	it("leaves the same session file as a turn pi drives itself", async () => {
		const whole = await createSession("whole");
		await whole.session.prompt("go");
		await whole.session.agent.waitForIdle();

		const split = await createSession("split");
		expect(await split.session.recordPrompt("go")).toBe(true);
		await driveStepped(split);

		// The loadout, two calls, then an answer, in both.
		expect(shape(persisted(whole.sessionManager))).toEqual([
			{ role: "system" },
			{ role: "user" },
			{ role: "assistant", content: ["toolCall", "toolCall"] },
			{ role: "toolResult", call: "call_1", error: false },
			{ role: "toolResult", call: "call_2", error: false },
			{ role: "assistant", content: ["text"] },
		]);
		expect(shape(persisted(split.sessionManager))).toEqual(shape(persisted(whole.sessionManager)));
		expect(split.ran).toEqual(whole.ran);
		expect(split.asked()).toBe(whole.asked());
	});

	it("keeps what the turn added to the system prompt across its steps", async () => {
		const extension = `export default p => p.on("before_agent_start", (event) => {
			event.systemPromptOptions.sections = { ...event.systemPromptOptions.sections, turn: "added" };
		});`;
		const whole = await createSession("whole-prompt", { extension });
		await whole.session.prompt("go");
		await whole.session.agent.waitForIdle();

		const split = await createSession("split-prompt", { extension });
		expect(await split.session.recordPrompt("go")).toBe(true);
		await driveStepped(split);

		// A step that dropped the turn's options would write a system message taking the section back.
		const systems = persisted(split.sessionManager).filter((m) => m.role === "system");
		expect(systems).toHaveLength(1);
		expect(systems[0].role === "system" && systems[0].sections?.turn).toContain("added");
		expect(shape(persisted(split.sessionManager))).toEqual(shape(persisted(whole.sessionManager)));
	});

	// Finishes the turn a session was opened on, the way pi does for an executor that asks.
	const RESUMING = `export default p => p.registerTurnExecutor((turn) => turn.run(), {
		resumeOnStart: true,
	});`;

	type ReopenOptions = {
		responses?: AssistantMessage[];
		extension?: string;
		systemPrompt?: string;
		fastRetry?: boolean;
		terminate?: boolean;
	};

	/** One model call, its tool calls, and the seal. */
	async function driveStep(harness: Harness): Promise<void> {
		const model = await harness.session.modelCall();
		const results: TurnToolCallOutcome[] = [];
		for (const call of model.toolCalls) {
			const result = await harness.session.runToolCall(call.id);
			if (result) results.push(result);
		}
		await harness.session.sealStep(results);
	}

	/**
	 * The session `name` opened again over its file, with memory rebuilt from it, the way the next
	 * activity opens it. Same name, so the same working directory and the same system prompt.
	 */
	async function reopen(first: Harness, name: string, options: ReopenOptions = {}) {
		const reuse = first.sessionManager.getSessionFile()!;
		const harness = await createSession(name, { ...options, reuse });
		const { messages } = harness.sessionManager.buildSessionProjection();
		harness.session.agent.state.messages = messages;
		return harness;
	}

	it("keeps a forced system prompt for a step taken in a reopened session", async () => {
		const extension = `export default p => p.on("before_agent_start", () => ({
			systemPrompt: "Exact prompt.",
		}));`;
		const first = await createSession("forced", { extension });
		await first.session.recordPrompt("go");
		await driveStep(first);
		expect(getCurrentSystemPrompt(first.requests[0].messages)).toBe("Exact prompt.");

		const again = await reopen(first, "forced", { extension });
		await again.session.modelCall();

		// A forced prompt is never recorded, so the turn's options are all that carry it.
		expect(getCurrentSystemPrompt(again.requests[0].messages)).toBe("Exact prompt.");
	});

	it("gives work after a finished turn the session's own options in a reopened session", async () => {
		const extension = `export default p => p.on("before_agent_start", () => ({
			systemPrompt: "Exact prompt.",
		}));`;
		const first = await createSession("finished-options", { extension });
		await first.session.recordPrompt("go");
		await driveStepped(first);

		// Not through a prompt, so no turn of its own records options. It is the session's work.
		const again = await reopen(first, "finished-options");
		again.session.agent.state.messages = [
			...again.session.agent.state.messages,
			{ role: "user", content: [{ type: "text", text: "more" }], timestamp: Date.now() },
		];
		await again.session.modelCall();

		expect(getCurrentSystemPrompt(again.requests[0].messages)).not.toBe("Exact prompt.");
	});

	it("keeps the turn's system prompt additions when a reopened session ends it", async () => {
		const extension = `export default p => p.on("before_agent_start", (event) => {
			const options = event.systemPromptOptions;
			options.sections = { ...options.sections, turn: "added" };
		});`;
		const first = await createSession("added", { extension });
		await first.session.recordPrompt("go");
		await driveStep(first);

		const call = { type: "toolCall" as const, id: "call_3", name: "dummy", arguments: {} };
		const answer = { type: "text" as const, text: "done" };
		const responses = [assistant([call], "toolUse"), assistant([answer])];
		const again = await reopen(first, "added", { extension: RESUMING, responses });
		await again.session.bindExtensions({});

		// The turn's later model calls rebuild the system prompt. Rebuilt from the session's own
		// options, it would write a system message that takes the section back.
		const systems = persisted(again.sessionManager).filter((m) => m.role === "system");
		expect(systems).toHaveLength(1);
		expect(systems[0].role === "system" && systems[0].sections?.turn).toContain("added");
		expect(again.asked()).toBe(2);
	});

	it("keeps a field the turn cleared when a reopened session ends the turn", async () => {
		const extension = `export default p => p.on("before_agent_start", (event) => {
			event.systemPromptOptions.customPrompt = undefined;
		});`;
		const systemPrompt = "The session's own prompt.";
		const first = await createSession("cleared", { extension, systemPrompt });
		await first.session.recordPrompt("go");
		await driveStep(first);

		const call = { type: "toolCall" as const, id: "call_3", name: "dummy", arguments: {} };
		const answer = { type: "text" as const, text: "done" };
		const responses = [assistant([call], "toolUse"), assistant([answer])];
		const reopened = { extension: RESUMING, systemPrompt, responses };
		const again = await reopen(first, "cleared", reopened);
		await again.session.bindExtensions({});

		// Restored as the session's own, the cleared prompt would come back in a system message.
		const systems = persisted(again.sessionManager).filter((m) => m.role === "system");
		expect(systems).toHaveLength(1);
		expect(JSON.stringify(systems)).not.toContain(systemPrompt);
		expect(again.asked()).toBe(2);
	});

	it("lets go of the options a tool call restored once the call is over", async () => {
		const extension = `export default p => p.on("before_agent_start", (event) => {
			event.systemPromptOptions.sections = { ...event.systemPromptOptions.sections, turn: "x" };
		});`;
		const first = await createSession("tool-only", { extension });
		await first.session.recordPrompt("go");
		const model = await first.session.modelCall();

		// A process that only runs the call, and whose session lives on after it.
		const again = await reopen(first, "tool-only", { extension });
		const internals = again.session as unknown as { _runSystemPromptOptions: unknown };
		await again.session.runToolCall(model.toolCalls[0].id);

		expect(internals._runSystemPromptOptions).toBeUndefined();
	});

	it("holds a call's result back until the step is sealed", async () => {
		const harness = await createSession("held");
		await harness.session.recordPrompt("go");

		const model = await harness.session.modelCall();
		expect(model.toolCalls.map((c) => c.id)).toEqual(["call_1", "call_2"]);
		expect(shape(persisted(harness.sessionManager))).toEqual([
			{ role: "system" },
			{ role: "user" },
			{ role: "assistant", content: ["toolCall", "toolCall"] },
		]);

		// Settle the second call first, the way concurrent dispatch does.
		const second = await harness.session.runToolCall("call_2");
		const first = await harness.session.runToolCall("call_1");
		expect(harness.ran).toEqual(["two", "one"]);

		// Nothing of theirs is in the file yet. Two writers appending as they finish would
		// branch the session tree, and the results would reach the model out of order.
		expect(persisted(harness.sessionManager).filter((m) => m.role === "toolResult")).toHaveLength(0);

		await harness.session.sealStep([first, second] as TurnToolCallOutcome[]);

		const results = persisted(harness.sessionManager).filter((m) => m.role === "toolResult");
		expect(results.map((m) => (m.role === "toolResult" ? m.toolCallId : ""))).toEqual(["call_1", "call_2"]);
	});

	it("does not ask the model again for a response it already recorded", async () => {
		const first = await createSession("recorded");
		await first.session.recordPrompt("go");
		const asked = await first.session.modelCall();

		// The retry lands somewhere else: a session opened over the same transcript, which is
		// what a driver whose record of the call was lost comes back as.
		const retry = await createSession("retry");
		retry.session.agent.state.messages = persisted(first.sessionManager);

		const replayed = await retry.session.modelCall();

		// Paying for a second response would also leave the first response's calls behind, and
		// the next resume would tell the model their outcome is unknown when nothing ran them.
		expect(retry.asked()).toBe(0);
		expect(replayed.toolCalls.map((c) => c.id)).toEqual(asked.toolCalls.map((c) => c.id));
	});

	it("keeps every failed attempt of a step in the transcript", async () => {
		// The file is the record of a stepped turn, and the session that made an attempt is gone
		// by the time anyone reads it. An attempt filtered out, or never written, did not happen.
		const failing = () => [{ ...assistant([{ type: "text", text: "" }]), stopReason: "error" as const }];

		const first = await createSession("failed-once", { responses: failing() });
		await first.session.recordPrompt("go");
		await first.session.modelCall();
		const file = first.sessionManager.getSessionFile()!;

		// The next attempt opens the same file, drops the error from memory, and fails again.
		const second = await createSession("failed-twice", { responses: failing(), reuse: file });
		second.session.agent.state.messages = persisted(first.sessionManager);
		second.session.prepareStep();
		await second.session.modelCall();

		// Trailing and consecutive, not just present: that is what the count reads.
		const tail = persisted(second.sessionManager).slice(-2);
		expect(tail.map((m) => m.role === "assistant" && m.stopReason)).toEqual(["error", "error"]);
	});

	it("runs a message queued during the last step instead of replaying the answer", async () => {
		const harness = await createSession("queued", {
			responses: [assistant([{ type: "text", text: "first" }]), assistant([{ type: "text", text: "second" }])],
		});
		await harness.session.recordPrompt("go");

		const seals: boolean[] = [];
		for (let step = 0; step < 6; step++) {
			await harness.session.modelCall();
			// Queued while the last step is still open, the way an extension's follow-up arrives.
			if (step === 0) {
				harness.session.agent.followUp({
					role: "user",
					content: [{ type: "text", text: "more" }],
					timestamp: Date.now(),
				});
			}
			const { done } = await harness.session.sealStep([]);
			seals.push(done);
			if (done) break;
		}

		expect(seals).toEqual([false, true]);
		expect(harness.asked()).toBe(2);
		expect(harness.session.agent.hasQueuedMessages()).toBe(false);
		expect(shape(persisted(harness.sessionManager)).filter((m) => m.role !== "system")).toEqual([
			{ role: "user" },
			{ role: "assistant", content: ["text"] },
			{ role: "user" },
			{ role: "assistant", content: ["text"] },
		]);
	});

	it("finishes a turn once when its seal runs again", async () => {
		const extension = `export default p => p.on("turn_end", () => {
			globalThis.steppedTurnEnds = (globalThis.steppedTurnEnds ?? 0) + 1;
		});`;
		delete testGlobals.steppedTurnEnds;
		const harness = await createSession("sealed-twice", {
			responses: [assistant([{ type: "text", text: "answer" }])],
			extension,
		});
		await harness.session.recordPrompt("go");
		await harness.session.modelCall();
		await harness.session.sealStep([]);
		await harness.session.sealStep([]);

		expect(testGlobals.steppedTurnEnds).toBe(1);
		delete testGlobals.steppedTurnEnds;
	});

	it("refuses a call or a seal that is late for its step", async () => {
		const call = { type: "toolCall" as const, id: "call_1", name: "dummy", arguments: { q: "one" } };
		const reused = { ...call, arguments: { q: "two" } };
		const responses = [assistant([call], "toolUse"), assistant([reused], "toolUse")];
		const harness = await createSession("late-for-step", { responses });
		await harness.session.recordPrompt("go");
		const first = await harness.session.modelCall();
		const result = await harness.session.runToolCall("call_1", { stepId: first.stepId });
		await harness.session.sealStep(result ? [result] : [], { stepId: first.stepId });

		// The next response reuses the call id with other arguments.
		const second = await harness.session.modelCall();
		expect(second.stepId).not.toBe(first.stepId);
		await expect(harness.session.runToolCall("call_1", { stepId: first.stepId })).rejects.toThrow("is over");
		await expect(harness.session.sealStep([], { stepId: first.stepId })).rejects.toThrow("is over");
		expect(harness.ran).toEqual(["one"]);

		// A rejected seal leaves the step open, so the right caller still closes it.
		const ran = await harness.session.runToolCall("call_1", { stepId: second.stepId });
		expect((await harness.session.sealStep(ran ? [ran] : [], { stepId: second.stepId })).done).toBe(false);
		expect(harness.ran).toEqual(["one", "two"]);
	});

	it("frees the session when a write at the end of a turn fails", async () => {
		const harness = await createSession("settle-write-fails", {
			responses: [assistant([{ type: "text", text: "answer" }])],
		});
		await harness.session.recordPrompt("go");
		await harness.session.modelCall();
		const internals = harness.session as unknown as {
			_recordedPromptOptionChanges(): Record<string, unknown>;
			_appendInternalEntry(customType: string, data: unknown): void;
		};
		internals._recordedPromptOptionChanges = () => ({ sections: {} });
		internals._appendInternalEntry = () => {
			throw new Error("not ours to write");
		};

		await expect(harness.session.sealStep([])).rejects.toThrow("not ours to write");
		expect(harness.session.isIdle).toBe(true);
	});

	it("keeps queued messages out of a step whose seal was refused", async () => {
		const harness = await createSession("refused-seal-queue");
		await harness.session.recordPrompt("go");
		await harness.session.modelCall();
		const internals = harness.session as unknown as { _pendingCustomMessages: unknown[] };
		internals._pendingCustomMessages = [
			{ role: "custom", customType: "note", content: "later", display: false, timestamp: Date.now() },
		];

		await expect(harness.session.sealStep([], { expectCalls: ["other"] })).rejects.toThrow("Cannot seal");
		// Written now, it would sit between the step's calls and their results.
		expect(harness.session.agent.state.messages.some((m) => m.role === "custom")).toBe(false);
		expect(internals._pendingCustomMessages).toHaveLength(1);
	});

	it("frees the session when the prompt's options can't be written", async () => {
		const harness = await createSession("options-write-fails");
		const internals = harness.session as unknown as { _recordTurnPromptOptions(): void };
		internals._recordTurnPromptOptions = () => {
			throw new Error("not ours to write");
		};
		await expect(harness.session.prompt("go")).rejects.toThrow("not ours to write");
		expect(harness.session.isIdle).toBe(true);
	});

	it("keeps the turn's options when a seal is refused", async () => {
		const harness = await createSession("refused-seal");
		await harness.session.recordPrompt("go");
		await harness.session.modelCall();
		const internals = harness.session as unknown as { _runSystemPromptOptions: unknown };
		const options = { cwd: "/kept" };
		internals._runSystemPromptOptions = options;

		await expect(harness.session.sealStep([], { expectCalls: ["other"] })).rejects.toThrow("Cannot seal");
		// The step is still open, and a refused seal is not the end of its turn.
		expect(internals._runSystemPromptOptions).toBe(options);
		expect(harness.session.isIdle).toBe(true);
	});

	it("refuses a seal while a tool call of the step is running", async () => {
		const harness = await createSession("seal-during-tool");
		await harness.session.recordPrompt("go");
		await harness.session.modelCall();
		let settled = 0;
		let attempt: Promise<unknown> | undefined;
		harness.session.subscribe((event) => {
			if (event.type === "agent_settled") settled++;
			if (event.type === "tool_execution_start") attempt = harness.session.sealStep([]);
		});
		const result = await harness.session.runToolCall("call_1");

		await expect(attempt).rejects.toThrow("already processing");
		// The refused seal didn't settle the run under the running call.
		expect(settled).toBe(0);
		expect(result).toBeDefined();
	});

	it("stops a running call when the driver's signal aborts", async () => {
		const harness = await createSession("signal-stops-call", { hang: true });
		await harness.session.recordPrompt("go");
		await harness.session.modelCall();
		const stop = new AbortController();
		harness.session.subscribe((event) => {
			// Once the tool is under way, as a cancellation arrives in practice.
			if (event.type === "tool_execution_start") setTimeout(() => stop.abort(), 20);
		});
		const result = await harness.session.runToolCall("call_1", { signal: stop.signal });

		// The tool ended and reported it, instead of running on after its driver gave up.
		expect(result?.message.isError).toBe(true);
		expect(JSON.stringify(result?.message.content)).toContain("stopped one");
		await expect(harness.session.runToolCall("call_2", { signal: stop.signal })).rejects.toThrow();
		expect(harness.ran).toEqual(["one"]);
	});

	it("stops a model call when the driver's signal aborts", async () => {
		const harness = await createSession("signal-stops-model", { hangModel: true });
		await harness.session.recordPrompt("go");
		const stop = new AbortController();
		harness.session.subscribe((event) => {
			// Once the provider request is under way, as a cancellation arrives in practice.
			if (event.type === "message_start" && event.message.role === "assistant") setTimeout(() => stop.abort(), 20);
		});
		const outcome = await harness.session.modelCall({ signal: stop.signal });

		// Reported like any aborted response, and recorded so the next attempt sees it.
		expect(outcome.ended).toBe(true);
		const last = persisted(harness.sessionManager).at(-1);
		expect(last?.role === "assistant" && last.stopReason).toBe("aborted");
		await harness.session.sealStep([]);
		expect(harness.session.isIdle).toBe(true);

		// The session takes the next turn.
		await harness.session.recordPrompt("again");
		await driveStepped(harness);
		expect(harness.ran).toEqual(["one", "two"]);
	});

	it("refuses a model call whose signal already aborted", async () => {
		const harness = await createSession("signal-aborted-model");
		await harness.session.recordPrompt("go");

		await expect(harness.session.modelCall({ signal: AbortSignal.abort() })).rejects.toThrow();
		// No provider request, and no run left open behind the refusal.
		expect(harness.asked()).toBe(0);
		expect(harness.session.isIdle).toBe(true);
		expect((await harness.session.modelCall()).toolCalls).toHaveLength(2);
	});

	it("replays what turn_end decided when its handler took the response out of the context", async () => {
		const extension = `export default p => p.on("turn_end", (event) => ({
			entries: [{ type: "context_edit", targetId: event.messageEntryId, replacement: null }],
		}));`;
		const responses = [assistant([{ type: "text", text: "answer" }])];
		const first = await createSession("omitted-by-hook", { responses, extension });
		await first.session.recordPrompt("go");
		await first.session.modelCall();
		const sealed = await first.session.sealStep([]);

		// Not a recovery. The replay must not ask for another model call.
		const again = await reopen(first, "omitted-by-hook-again", { responses, extension });
		expect(await again.session.sealStep([])).toEqual(sealed);
		expect(again.asked()).toBe(0);
	});

	it("keeps a turn its tools stopped over when the seal runs again in a reopened session", async () => {
		const call = { type: "toolCall" as const, id: "call_1", name: "dummy", arguments: { q: "one" } };
		const responses = [assistant([call], "toolUse"), assistant([{ type: "text", text: "never" }])];
		const first = await createSession("stopped-by-tool", { responses, terminate: true });
		await first.session.recordPrompt("go");
		await first.session.modelCall();
		const result = await first.session.runToolCall("call_1");
		expect((await first.session.sealStep(result ? [result] : [], { expectCalls: ["call_1"] })).done).toBe(true);

		// The transcript ends on the result. Without the marker it reads as a turn with more to do.
		const again = await reopen(first, "stopped-by-tool-again", { responses, terminate: true });
		expect(again.session.prepareStep()).toBe(false);
		expect((await again.session.sealStep([], { expectCalls: ["call_1"] })).done).toBe(true);
		expect(again.asked()).toBe(0);
	});

	it("finishes a turn once when its seal runs again in a reopened session", async () => {
		// The retry a durable driver makes: the seal's unit of work died after the boundary
		// committed, and the next attempt opens the file in a process that remembers nothing.
		const extension = `export default p => p.on("turn_end", () => {
			globalThis.reopenedTurnEnds = (globalThis.reopenedTurnEnds ?? 0) + 1;
			return { entries: [{ type: "custom", customType: "turn-note", data: {} }] };
		});`;
		delete testGlobals.reopenedTurnEnds;
		const responses = [assistant([{ type: "text", text: "answer" }])];
		const first = await createSession("sealed-reopened", { responses, extension });
		await first.session.recordPrompt("go");
		await first.session.modelCall();
		await first.session.sealStep([]);
		const file = first.sessionManager.getSessionFile()!;

		const reopened = { responses, extension, reuse: file };
		const again = await createSession("sealed-reopened-again", reopened);
		again.session.agent.state.messages = again.sessionManager.buildSessionProjection().messages;
		await again.session.sealStep([]);

		expect(testGlobals.reopenedTurnEnds).toBe(1);
		delete testGlobals.reopenedTurnEnds;
		const notes = SessionManager.open(file)
			.getBranch()
			.filter((entry) => entry.type === "custom" && entry.customType === "turn-note");
		expect(notes).toHaveLength(1);
	});

	it("runs agent_before_settle once when a reopened final seal runs again", async () => {
		const extension = `export default p => p.on("agent_before_settle", () => {
			globalThis.reopenedSettles = (globalThis.reopenedSettles ?? 0) + 1;
			return { entries: [{ type: "custom", customType: "settle-note", data: {} }] };
		});`;
		delete testGlobals.reopenedSettles;
		const responses = [assistant([{ type: "text", text: "answer" }])];
		const first = await createSession("settled-reopened", { responses, extension });
		await first.session.recordPrompt("go");
		await first.session.modelCall();
		expect((await first.session.sealStep([])).done).toBe(true);

		const again = await reopen(first, "settled-reopened-again", { responses, extension });
		expect((await again.session.sealStep([])).done).toBe(true);

		expect(testGlobals.reopenedSettles).toBe(1);
		delete testGlobals.reopenedSettles;
		const notes = SessionManager.open(first.sessionManager.getSessionFile()!)
			.getBranch()
			.filter((entry) => entry.type === "custom" && entry.customType === "settle-note");
		expect(notes).toHaveLength(1);
	});

	it("continues as the first seal decided when a reopened seal runs again", async () => {
		const extension = `export default p => {
			p.on("turn_end", () => {
				if (globalThis.reopenedContinued) return;
				globalThis.reopenedContinued = true;
				const next = { type: "custom_message", customType: "n", content: "on", display: false };
				return { entries: [next], continue: true };
			});
			p.on("agent_before_settle", () => {
				globalThis.reopenedSettles = (globalThis.reopenedSettles ?? 0) + 1;
			});
		};`;
		delete testGlobals.reopenedContinued;
		delete testGlobals.reopenedSettles;
		const responses = [assistant([{ type: "text", text: "first" }])];
		const first = await createSession("continued", { responses, extension });
		await first.session.recordPrompt("go");
		await first.session.modelCall();
		expect((await first.session.sealStep([])).done).toBe(false);

		const again = await reopen(first, "continued", { responses, extension });
		const { done } = await again.session.sealStep([]);

		// Ending here would leave the message the boundary committed unanswered.
		expect(done).toBe(false);
		expect(testGlobals.reopenedSettles).toBeUndefined();
		delete testGlobals.reopenedContinued;
		delete testGlobals.reopenedSettles;
	});

	async function driveAnswers(name: string, extension: string): Promise<{ harness: Harness; seals: boolean[] }> {
		const harness = await createSession(name, {
			responses: [assistant([{ type: "text", text: "first" }]), assistant([{ type: "text", text: "second" }])],
			extension,
		});
		await harness.session.recordPrompt("go");
		const seals: boolean[] = [];
		for (let step = 0; step < 6; step++) {
			await harness.session.modelCall();
			const { done } = await harness.session.sealStep([]);
			seals.push(done);
			if (done) break;
		}
		return { harness, seals };
	}

	it("keeps going when turn_end asks to continue and leaves something to run", async () => {
		const { harness, seals } = await driveAnswers(
			"continue-runnable",
			`export default p => p.on("turn_end", () => {
				if (globalThis.steppedContinued) return;
				globalThis.steppedContinued = true;
				return {
					entries: [{ type: "custom_message", customType: "next-work", content: "keep going", display: false }],
					continue: true,
				};
			});`,
		);
		delete testGlobals.steppedContinued;

		expect(seals).toEqual([false, true]);
		expect(harness.asked()).toBe(2);
	});

	it("ends at a settled answer even when turn_end asks to continue", async () => {
		// Continuing from the answer itself would take a context-only turn, which a step cannot make.
		const { harness, seals } = await driveAnswers(
			"continue-settled",
			`export default p => p.on("turn_end", () => {
				if (globalThis.steppedContinued) return;
				globalThis.steppedContinued = true;
				return { continue: true };
			});`,
		);
		delete testGlobals.steppedContinued;

		expect(seals).toEqual([true]);
		expect(harness.asked()).toBe(1);
	});

	it("emits agent_before_settle when a stepped turn ends, without honouring its continuation", async () => {
		const { harness, seals } = await driveAnswers(
			"before-settle",
			`export default p => p.on("agent_before_settle", () => {
				globalThis.steppedBeforeSettle = (globalThis.steppedBeforeSettle ?? 0) + 1;
				return { continue: true };
			});`,
		);
		const emitted = testGlobals.steppedBeforeSettle;
		delete testGlobals.steppedBeforeSettle;

		expect(emitted).toBe(1);
		expect(seals).toEqual([true]);
		expect(harness.asked()).toBe(1);
	});

	it("answers a retry seal that lost its answer the same way in a reopened session", async () => {
		const failed = assistant([{ type: "text", text: "" }], "error");
		failed.errorMessage = "overloaded";
		const responses = [failed, assistant([{ type: "text", text: "answer" }])];
		const first = await createSession("lost-retry", { responses, fastRetry: true });
		await first.session.recordPrompt("go");
		const { stepId } = await first.session.modelCall();
		const decided = await first.session.sealStep([], { retryAttempt: 0, stepId });
		expect(decided).toMatchObject({ done: false, retryAttempt: 1 });

		// The failed response is out of the model's context, so there is no step left to seal.
		// The driver still holds the count it had before the lost answer.
		const again = await reopen(first, "lost-retry-again", { responses, fastRetry: true });
		expect(await again.session.sealStep([], { retryAttempt: 0, stepId })).toEqual(decided);
		// A caller late for another step is refused, not handed this step's decision.
		await expect(again.session.sealStep([], { retryAttempt: 0, stepId: "older" })).rejects.toThrow("is over");
	});

	it("is busy while a reopened session runs a tool call", async () => {
		const first = await createSession("busy-tool");
		await first.session.recordPrompt("go");
		await first.session.modelCall();

		const again = await reopen(first, "busy-tool-again");
		const seen: Array<boolean | "busy"> = [];
		again.session.subscribe((event) => {
			if (event.type === "tool_execution_start") seen.push(again.session.prepareStep());
		});
		await again.session.runToolCall("call_1");

		// Settling now would record an unknown outcome for a call that is still running.
		expect(seen).toEqual(["busy"]);
		expect(again.session.isIdle).toBe(true);
	});

	it("gives a step that got an answer a fresh retry budget", async () => {
		// The count is carried across activities by the caller, and a session rebuilt per activity
		// never sees the message event that resets it in a live one. Without the reset here, three
		// failures anywhere in a turn end the next step that fails, however well it went between.
		const harness = await createSession("budget-reset");
		harness.session.agent.state.messages = [
			{ role: "user", content: [{ type: "text", text: "go" }], timestamp: Date.now() },
			assistant([{ type: "text", text: "an answer" }]),
		];

		const result = await harness.session.sealStep([], { retryAttempt: 3, overflowRecoveryAttempted: true });

		// The compact-and-retry a turn gets ends with an answer too.
		expect(result).toMatchObject({ retryAttempt: 0, overflowRecoveryAttempted: false });
	});

	it("refuses a second model call while the step is still open", async () => {
		const harness = await createSession("busy");
		await harness.session.recordPrompt("go");
		await harness.session.modelCall();

		// Answering "the response ended the run" would have the caller seal a step it never
		// opened, which closes the previous one a second time.
		await expect(harness.session.modelCall()).rejects.toThrow(/already processing/);
	});

	it("seals against the message the model call left, not the one it made itself", async () => {
		const asking = await createSession("post-run-model");
		await asking.session.recordPrompt("go");
		const first = await asking.session.modelCall();
		const results: TurnToolCallOutcome[] = [];
		for (const call of first.toolCalls) {
			const result = await asking.session.runToolCall(call.id);
			if (result) results.push(result);
		}
		await asking.session.sealStep(results);
		// The last step asks for no tools, so nothing but the post-run pass can keep the turn going.
		await asking.session.modelCall();

		// The seal is its own activity, so on the worker path it opens a session that never saw
		// the model call happen. That is the whole point of the split, and it is what makes the
		// post-run pass read the transcript instead of its own memory.
		const sealing = await createSession("post-run-seal");
		sealing.session.agent.state.messages = persisted(asking.sessionManager);
		// A queued message is the cheapest thing the pass answers for. A provider error that wants
		// a retry, and a full context that wants a compaction, reach the same code the same way.
		sealing.session.agent.followUp({
			role: "user",
			content: [{ type: "text", text: "and then?" }],
			timestamp: Date.now(),
		});

		const { done } = await sealing.session.sealStep([]);

		expect(done).toBe(false);
	});

	it("does not run a call twice when the step is driven again", async () => {
		const harness = await createSession("twice");
		await harness.session.recordPrompt("go");
		await harness.session.modelCall();

		const once = await harness.session.runToolCall("call_1");
		await harness.session.sealStep([once] as TurnToolCallOutcome[]);
		const again = await harness.session.runToolCall("call_1");

		// The transcript already answers for it, so the tool is left alone. Re-running a tool
		// whose effect already happened is the worse failure for a coding agent.
		expect(again).toBeUndefined();
		expect(harness.ran).toEqual(["one"]);
	});
});
