import { existsSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, type AgentMessage } from "@earendil-works/pi-agent-core";
import { transformMessages } from "@earendil-works/pi-ai/api/transform-messages";
import {
	type AssistantMessage,
	type AssistantMessageEvent,
	EventStream,
	getModel,
	type Message,
	type ToolResultMessage,
	type UserMessage,
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

function user(text: string): UserMessage {
	return { role: "user", content: [{ type: "text", text }], timestamp: Date.now() };
}

const call = (id: string) => ({ type: "toolCall" as const, id, name: "do", arguments: {} });
const answer = (text: string) => assistant([{ type: "text", text }]);

/** Every tool result must follow the assistant message that asked for it, exactly once. */
function assertValidToolPairing(messages: AgentMessage[]): void {
	const model = getModel("anthropic", "claude-sonnet-4-5")!;
	const payload = transformMessages(messages as Message[], model);
	const seen = new Set<string>();
	for (let i = 0; i < payload.length; i++) {
		const message = payload[i];
		if (message.role !== "toolResult") {
			continue;
		}
		expect(seen.has(message.toolCallId)).toBe(false);
		seen.add(message.toolCallId);

		let owner = i - 1;
		while (owner >= 0 && payload[owner].role === "toolResult") {
			owner--;
		}
		const asking = payload[owner];
		expect(asking?.role).toBe("assistant");
		const calls = asking?.role === "assistant" ? asking.content : [];
		const ids = calls.filter((b) => b.type === "toolCall").map((b) => b.id);
		expect(ids).toContain(message.toolCallId);
	}
}

describe("AgentSession: settling an interrupted turn", () => {
	let session: AgentSession;
	let sessionManager: SessionManager;
	let tempDir: string;
	let modelCalls: number;

	beforeEach(() => {
		const unique = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
		tempDir = join(tmpdir(), `pi-resume-test-${unique}`);
		mkdirSync(tempDir, { recursive: true });
		modelCalls = 0;
	});

	afterEach(() => {
		if (session) session.dispose();
		if (tempDir && existsSync(tempDir)) rmSync(tempDir, { recursive: true });
	});

	/** A session whose model answers from `replies`, repeating the last one. */
	async function createSession(replies: AssistantMessage[] = [answer("all handled")]): Promise<AgentSession> {
		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model, systemPrompt: "Test", tools: [] },
			streamFn: () => {
				const stream = new MockAssistantStream();
				const message = replies[Math.min(modelCalls, replies.length - 1)];
				modelCalls++;
				queueMicrotask(() => {
					stream.push({ type: "done", reason: "stop", message });
				});
				return stream;
			},
		});

		sessionManager = SessionManager.create(tempDir, join(tempDir, "sessions"));
		const settingsManager = SettingsManager.create(tempDir, tempDir);
		// The retry path is under test, not the wait in front of it.
		settingsManager.applyOverrides({ retry: { enabled: true, maxRetries: 3, baseDelayMs: 1 } });
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

	/** Messages as the session file holds them, so persistence is covered too. */
	function persisted(): AgentMessage[] {
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

	/** Seed both memory and the file, the way a crashed run leaves them. */
	function seed(messages: (UserMessage | AssistantMessage | ToolResultMessage)[]): void {
		session.agent.state.messages = messages;
		for (const message of messages) {
			sessionManager.appendMessage(message);
		}
	}

	it("settles the calls of an interrupted turn and drives it to completion", async () => {
		await createSession();
		seed([user("do two things"), assistant([call("hang-1"), call("hang-2")], "toolUse")]);

		expect(await session.resumeInterruptedTurn()).toBe(true);
		expect(session.isIdle).toBe(true);

		const messages = session.agent.state.messages;

		// The prompt was not added again.
		expect(messages.filter((m) => m.role === "user").length).toBe(1);

		// Both calls are settled, and the turn produced a final answer.
		const settled = messages.filter((m) => m.role === "toolResult") as ToolResultMessage[];
		expect(settled.map((m) => m.toolCallId)).toEqual(["hang-1", "hang-2"]);
		expect(settled.every((m) => m.isError)).toBe(true);
		expect(findDanglingToolCalls(messages)).toEqual([]);
		const last = messages[messages.length - 1];
		expect(last.role === "assistant" && last.content[0]).toMatchObject({ type: "text", text: "all handled" });

		// The settled results reached the session file, not just memory.
		const written = persisted().filter((m) => m.role === "toolResult") as ToolResultMessage[];
		expect(written.map((m) => m.toolCallId)).toEqual(["hang-1", "hang-2"]);

		// The repaired transcript is a payload a provider accepts.
		assertValidToolPairing(messages);
	});

	it("keeps a result that landed before the interruption", async () => {
		await createSession();
		seed([
			user("do two things"),
			assistant([call("done-1"), call("hang-1")], "toolUse"),
			toolResult("done-1", "did done-1", false),
		]);

		expect(await session.resumeInterruptedTurn()).toBe(true);

		const messages = session.agent.state.messages;
		const kept = messages.find((m) => m.role === "toolResult" && m.toolCallId === "done-1") as ToolResultMessage;
		expect(kept.isError).toBe(false);
		expect(kept.content?.[0]).toMatchObject({ type: "text", text: "did done-1" });

		// The turn finished, and the payload stays valid for the provider.
		expect(messages[messages.length - 1].role).toBe("assistant");
		assertValidToolPairing(messages);
	});

	it("resumes after an aborted assistant message", async () => {
		await createSession();
		seed([user("go"), assistant([{ type: "text", text: "" }], "aborted")]);

		expect(await session.resumeInterruptedTurn()).toBe(true);

		const messages = session.agent.state.messages;
		expect(messages.filter((m) => m.role === "user").length).toBe(1);
		const last = messages[messages.length - 1];
		expect(last.role === "assistant" && last.stopReason).toBe("stop");
	});

	it("leaves a finished turn alone", async () => {
		await createSession();
		seed([user("hi"), answer("done")]);

		expect(session.prepareStep()).toBe(false);
		expect(await session.resumeInterruptedTurn()).toBe(false);
		expect(modelCalls).toBe(0);
		expect(session.agent.state.messages.length).toBe(2);
	});

	it("settles a call once, however many times the driver asks", async () => {
		await createSession();
		seed([user("go"), assistant([call("hang-1")], "toolUse")]);

		expect(session.prepareStep()).toBe(true);
		expect(session.prepareStep()).toBe(true);
		expect(session.agent.state.messages.filter((m) => m.role === "toolResult").length).toBe(1);
	});
});
