// A turn driven as a model call, its tool calls and a seal, against the session it writes to.
// The loop's own tests cover the parts; what is pinned here is the session file, which is what
// an outside driver actually keeps: the same turn split three ways has to leave the same
// transcript, and a call's result must not reach the file before the step is sealed.
import { existsSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, type AgentMessage, type AgentTool, type TurnToolCallOutcome } from "@earendil-works/pi-agent-core";
import { type AssistantMessage, type AssistantMessageEvent, EventStream, getModel } from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AgentSession } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";
import { createTestResourceLoader } from "./utilities.ts";

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

	async function createSession(name: string): Promise<Harness> {
		const ran: string[] = [];
		const responses = script();
		let count = 0;
		const tool: AgentTool<typeof toolSchema, { q: string }> = {
			name: "dummy",
			label: "Dummy",
			description: "Records what it was asked",
			parameters: toolSchema,
			async execute(_id, params) {
				ran.push(params.q);
				return { content: [{ type: "text", text: `did ${params.q}` }], details: { q: params.q } };
			},
		};
		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model, systemPrompt: "Test", tools: [] },
			streamFn: () => {
				const stream = new MockAssistantStream();
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
		const sessionManager = SessionManager.create(sessionDir, join(sessionDir, "sessions"));
		const settingsManager = SettingsManager.create(sessionDir, sessionDir);
		const authStorage = AuthStorage.create(join(sessionDir, "auth.json"));
		const modelRegistry = await createModelRegistry(authStorage, sessionDir);
		await authStorage.modify("anthropic", async () => ({ type: "api_key", key: "test-key" }));

		const session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: sessionDir,
			modelRuntime: getModelRuntime(modelRegistry),
			resourceLoader: createTestResourceLoader(),
			baseToolsOverride: { dummy: tool },
		});
		built.push(session);
		return { session, sessionManager, ran, asked: () => count };
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

		// Two calls, then an answer, in both.
		expect(shape(persisted(whole.sessionManager))).toEqual([
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

	it("holds a call's result back until the step is sealed", async () => {
		const harness = await createSession("held");
		await harness.session.recordPrompt("go");

		const model = await harness.session.modelCall();
		expect(model.toolCalls.map((c) => c.id)).toEqual(["call_1", "call_2"]);
		expect(shape(persisted(harness.sessionManager))).toEqual([
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
		expect(replayed.replayed).toBe(true);
		expect(replayed.toolCalls.map((c) => c.id)).toEqual(asked.toolCalls.map((c) => c.id));
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
