import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { Agent } from "@earendil-works/pi-agent-core";
import { type AssistantMessage, type AssistantMessageEvent, EventStream, getModel } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AgentSession } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import type { TurnSteps } from "../src/core/extensions/index.ts";
import { discoverAndLoadExtensions } from "../src/core/extensions/loader.ts";
import { ExtensionRunner } from "../src/core/extensions/runner.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createInMemoryModelRegistry, createModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";
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

/** For a test about which executor wins, where the turn itself is beside the point. */
const noSteps: TurnSteps = {
	record: async () => {},
	interrupted: () => false,
	modelCall: async () => ({ toolCalls: [], sequential: false, ended: true, replayed: false }),
	runToolCall: async () => undefined,
	sealStep: async () => ({ done: true }),
};

function assistant(text: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "mock",
		usage,
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

describe("turn executor", () => {
	let tempDir: string;
	let extensionsDir: string;
	let session: AgentSession | undefined;
	let modelCalls: number;

	beforeEach(() => {
		tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-turn-executor-"));
		extensionsDir = path.join(tempDir, "extensions");
		fs.mkdirSync(extensionsDir);
		modelCalls = 0;
		delete (globalThis as any).turnsSeen;
	});

	afterEach(() => {
		if (session) session.dispose();
		session = undefined;
		delete (globalThis as any).turnsSeen;
		fs.rmSync(tempDir, { recursive: true, force: true });
	});

	async function loadExtensions(...sources: string[]) {
		fs.rmSync(extensionsDir, { recursive: true, force: true });
		fs.mkdirSync(extensionsDir);
		for (let i = 0; i < sources.length; i++) {
			fs.writeFileSync(path.join(extensionsDir, `e${i}.ts`), sources[i]);
		}
		return await discoverAndLoadExtensions([], tempDir, tempDir);
	}

	async function createRunner(...sources: string[]) {
		const result = await loadExtensions(...sources);
		const modelRegistry = await createInMemoryModelRegistry(AuthStorage.inMemory());
		return new ExtensionRunner(result.extensions, result.runtime, tempDir, SessionManager.inMemory(), modelRegistry);
	}

	async function createSession(...sources: string[]): Promise<AgentSession> {
		const extensionsResult = await loadExtensions(...sources);
		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model, systemPrompt: "Test", tools: [] },
			streamFn: () => {
				const stream = new MockAssistantStream();
				modelCalls++;
				queueMicrotask(() => {
					stream.push({ type: "done", reason: "stop", message: assistant("answer") });
				});
				return stream;
			},
		});

		const sessionManager = SessionManager.create(tempDir, path.join(tempDir, "sessions"));
		const settingsManager = SettingsManager.create(tempDir, tempDir);
		const authStorage = AuthStorage.create(path.join(tempDir, "auth.json"));
		const modelRegistry = await createModelRegistry(authStorage, tempDir);
		await authStorage.modify("anthropic", async () => ({ type: "api_key", key: "test-key" }));

		session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: tempDir,
			modelRuntime: getModelRuntime(modelRegistry),
			resourceLoader: createTestResourceLoader({ extensionsResult }),
		});
		return session;
	}

	it("has no executor until an extension registers one", async () => {
		const runner = await createRunner(`export default p => {};`);
		expect(runner.getTurnExecutor()).toBeUndefined();
	});

	it("gives the first registration to the session when two extensions want it", async () => {
		const runner = await createRunner(
			`export default p => p.registerTurnExecutor(async t => { globalThis.turnsSeen = "first"; await t.run(); });`,
			`export default p => p.registerTurnExecutor(async t => { globalThis.turnsSeen = "second"; await t.run(); });`,
		);
		const registered = runner.getTurnExecutor();
		expect(registered).toBeDefined();
		await registered?.executor({ sessionId: "s", run: async () => {}, steps: noSteps });
		expect((globalThis as any).turnsSeen).toBe("first");
	});

	it("runs the same turn through the executor as without one", async () => {
		await createSession(
			`export default p => p.registerTurnExecutor(async turn => {
				globalThis.turnsSeen = (globalThis.turnsSeen ?? []).concat(turn.sessionId);
				await turn.run();
			});`,
		);

		await (session as AgentSession).prompt("go");
		await (session as AgentSession).waitForIdle();

		// The executor was handed the turn, once, with the session it belongs to.
		const seen = (globalThis as any).turnsSeen as string[];
		expect(seen.length).toBe(1);
		expect(seen[0]).toBe((session as AgentSession).sessionManager.getSessionId());

		// And the turn itself is the ordinary one: prompt, one model call, the answer.
		expect(modelCalls).toBe(1);
		const messages = (session as AgentSession).agent.state.messages;
		expect(messages.map((m) => m.role)).toEqual(["user", "assistant"]);
		const last = messages[messages.length - 1];
		expect(last.role === "assistant" && last.content[0]).toMatchObject({ type: "text", text: "answer" });
	});

	it("runs the same turn through an executor that drives the steps itself", async () => {
		await createSession(
			`export default p => p.registerTurnExecutor(async turn => {
				await turn.steps.record();
				for (;;) {
					const model = await turn.steps.modelCall();
					const results = [];
					if (!model.ended) {
						for (const call of model.toolCalls) {
							const result = await turn.steps.runToolCall(call.id);
							if (result) results.push(result);
						}
					}
					globalThis.turnsSeen = (globalThis.turnsSeen ?? []).concat("step");
					const sealed = await turn.steps.sealStep(results);
					if (sealed.done) return;
				}
			});`,
		);

		await (session as AgentSession).prompt("go");
		await (session as AgentSession).waitForIdle();

		// One step, and the same turn as the one pi would have run: the prompt reached the
		// transcript without a model call of its own, and the answer is where run() leaves it.
		expect((globalThis as any).turnsSeen).toEqual(["step"]);
		expect(modelCalls).toBe(1);
		const messages = (session as AgentSession).agent.state.messages;
		expect(messages.map((m) => m.role)).toEqual(["user", "assistant"]);
		const last = messages[messages.length - 1];
		expect(last.role === "assistant" && last.content[0]).toMatchObject({ type: "text", text: "answer" });
	});

	it("hands an unfinished turn to an executor that asked for it, when the session opens", async () => {
		await createSession(
			`export default p => p.registerTurnExecutor(
				async turn => { globalThis.turnsSeen = (globalThis.turnsSeen ?? []).concat("resume"); await turn.run(); },
				{ resumeOnStart: true },
			);`,
		);
		const s = session as AgentSession;

		// The transcript a crash mid tool call leaves behind.
		s.agent.state.messages = [
			{ role: "user", content: [{ type: "text", text: "go" }], timestamp: Date.now() },
			{
				role: "assistant",
				content: [{ type: "toolCall", id: "hang-1", name: "do", arguments: {} }],
				api: "anthropic-messages",
				provider: "anthropic",
				model: "mock",
				usage,
				stopReason: "toolUse",
				timestamp: Date.now(),
			},
		];

		await s.bindExtensions({});
		await s.waitForIdle();

		expect((globalThis as any).turnsSeen).toEqual(["resume"]);
		// The dangling call was settled and the turn ran on to an answer, without a second prompt.
		const messages = s.agent.state.messages;
		expect(messages.filter((m) => m.role === "user").length).toBe(1);
		expect(messages.filter((m) => m.role === "toolResult").length).toBe(1);
		expect(messages[messages.length - 1].role).toBe("assistant");
	});

	it("leaves a finished session alone on start", async () => {
		await createSession(
			`export default p => p.registerTurnExecutor(
				async turn => { globalThis.turnsSeen = "ran"; await turn.run(); },
				{ resumeOnStart: true },
			);`,
		);
		const s = session as AgentSession;
		s.agent.state.messages = [
			{ role: "user", content: [{ type: "text", text: "hi" }], timestamp: Date.now() },
			{
				role: "assistant",
				content: [{ type: "text", text: "done" }],
				api: "anthropic-messages",
				provider: "anthropic",
				model: "mock",
				usage,
				stopReason: "stop",
				timestamp: Date.now(),
			},
		];

		await s.bindExtensions({});

		expect((globalThis as any).turnsSeen).toBeUndefined();
		expect(modelCalls).toBe(0);
	});

	it("leaves the turn unrun when the executor never runs it", async () => {
		await createSession(
			`export default p => p.registerTurnExecutor(async () => { globalThis.turnsSeen = "held"; });`,
		);

		await (session as AgentSession).prompt("go");

		// Holding a turn holds the whole turn, prompt included: run() is what records it.
		expect((globalThis as any).turnsSeen).toBe("held");
		expect(modelCalls).toBe(0);
		expect((session as AgentSession).agent.state.messages).toEqual([]);
	});
});
