import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, type AgentMessage } from "@earendil-works/pi-agent-core";
import {
	type AssistantMessage,
	type AssistantMessageEvent,
	EventStream,
	getModel,
	type ToolResultMessage,
} from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AgentSession } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { findDanglingToolCalls, SessionManager } from "../src/core/session-manager.ts";
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

function toolResult(toolCallId: string, text: string, isError: boolean): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId,
		toolName: "do",
		content: [{ type: "text", text }],
		details: {},
		isError,
		timestamp: Date.now(),
	};
}

function user(text: string): AgentMessage {
	return { role: "user", content: [{ type: "text", text }], timestamp: Date.now() };
}

describe("findDanglingToolCalls", () => {
	it("returns nothing for an empty transcript", () => {
		expect(findDanglingToolCalls([])).toEqual([]);
	});

	it("returns nothing when every tool call has a result", () => {
		const messages: AgentMessage[] = [
			user("go"),
			assistant([{ type: "toolCall", id: "t1", name: "do", arguments: {} }], "toolUse"),
			toolResult("t1", "done", false),
			assistant([{ type: "text", text: "all set" }]),
		];
		expect(findDanglingToolCalls(messages)).toEqual([]);
	});

	it("finds a single dangling tool call", () => {
		const messages: AgentMessage[] = [
			user("go"),
			assistant([{ type: "toolCall", id: "t1", name: "do", arguments: {} }], "toolUse"),
		];
		expect(findDanglingToolCalls(messages).map((c) => c.id)).toEqual(["t1"]);
	});

	it("finds multiple dangling calls across messages, ignoring resolved ones", () => {
		const messages: AgentMessage[] = [
			user("go"),
			assistant(
				[
					{ type: "toolCall", id: "t1", name: "do", arguments: {} },
					{ type: "toolCall", id: "t2", name: "do", arguments: {} },
				],
				"toolUse",
			),
			toolResult("t1", "done", false),
			assistant([{ type: "toolCall", id: "t3", name: "do", arguments: {} }], "toolUse"),
		];
		expect(findDanglingToolCalls(messages).map((c) => c.id)).toEqual(["t2", "t3"]);
	});
});

describe("AgentSession.resumeInterruptedTurn", () => {
	let session: AgentSession;
	let tempDir: string;

	beforeEach(() => {
		tempDir = join(tmpdir(), `pi-resume-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(tempDir, { recursive: true });
	});

	afterEach(() => {
		if (session) session.dispose();
		if (tempDir && existsSync(tempDir)) rmSync(tempDir, { recursive: true });
	});

	async function createSession(): Promise<AgentSession> {
		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model, systemPrompt: "Test", tools: [] },
			streamFn: () => {
				const stream = new MockAssistantStream();
				queueMicrotask(() => {
					stream.push({
						type: "done",
						reason: "stop",
						message: assistant([{ type: "text", text: "all handled" }]),
					});
				});
				return stream;
			},
		});

		const sessionManager = SessionManager.inMemory();
		const settingsManager = SettingsManager.create(tempDir, tempDir);
		const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
		const modelRegistry = await createModelRegistry(authStorage, tempDir);
		await authStorage.modify("anthropic", async () => ({ type: "api_key", key: "test-key" }));

		session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: tempDir,
			modelRuntime: getModelRuntime(modelRegistry),
			resourceLoader: createTestResourceLoader(),
		});
		return session;
	}

	it("repairs a dangling tool call and drives the turn to completion", async () => {
		await createSession();

		// A crash left one tool call answered and one dangling.
		session.agent.state.messages = [
			user("do two things"),
			assistant(
				[
					{ type: "toolCall", id: "done-1", name: "do", arguments: {} },
					{ type: "toolCall", id: "hang-1", name: "do", arguments: {} },
				],
				"toolUse",
			),
			toolResult("done-1", "did done-1", false),
		];

		const drove = await session.resumeInterruptedTurn();
		expect(drove).toBe(true);
		expect(session.isIdle).toBe(true);

		const messages = session.agent.state.messages;

		// The prompt was not re-added.
		expect(messages.filter((m) => m.role === "user").length).toBe(1);

		// The completed tool result is kept as-is.
		const completed = messages.find((m) => m.role === "toolResult" && m.toolCallId === "done-1") as
			| ToolResultMessage
			| undefined;
		expect(completed?.isError).toBe(false);
		expect(completed?.content?.[0]).toMatchObject({ type: "text", text: "did done-1" });

		// The dangling call is now settled as a failed result.
		const repaired = messages.find((m) => m.role === "toolResult" && m.toolCallId === "hang-1") as
			| ToolResultMessage
			| undefined;
		expect(repaired?.isError).toBe(true);

		// No tool call is left dangling, and the turn produced a final answer.
		expect(findDanglingToolCalls(messages)).toEqual([]);
		const last = messages[messages.length - 1];
		expect(last.role).toBe("assistant");
		expect(last.role === "assistant" && last.content[0]).toMatchObject({ type: "text", text: "all handled" });
	});

	it("is a no-op when the turn already finished", async () => {
		await createSession();
		session.agent.state.messages = [user("hi"), assistant([{ type: "text", text: "done" }])];

		const drove = await session.resumeInterruptedTurn();
		expect(drove).toBe(false);
		expect(session.agent.state.messages.length).toBe(2);
	});
});
