import { existsSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, type AgentMessage } from "@earendil-works/pi-agent-core";
import { transformMessages } from "@earendil-works/pi-ai/api/transform-messages";
import {
	type AssistantMessage,
	type AssistantMessageEvent,
	contentText,
	EventStream,
	fauxAssistantMessage,
	getModel,
	type ToolResultMessage,
	type UserMessage,
} from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AgentSession } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { type CustomMessage, convertToLlm } from "../src/core/messages.ts";
import { findDanglingToolCalls, SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import type { ExtensionFactory } from "../src/index.ts";
import { createModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";
import { createHarness, getMessageText, getUserTexts } from "./suite/harness.ts";
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
const note = (text: string): CustomMessage => ({
	role: "custom",
	customType: "note",
	content: text,
	display: true,
	timestamp: Date.now(),
});
const isUnknownOutcome = (message: AgentMessage | undefined) =>
	message?.role === "toolResult" &&
	message.isError &&
	contentText(message.content, "").startsWith("The outcome of this tool call is unknown.");
const answer = (text: string) => assistant([{ type: "text", text }]);

/** Every tool result must follow the assistant message that asked for it, exactly once. */
function assertValidToolPairing(messages: AgentMessage[]): void {
	const model = getModel("anthropic", "claude-sonnet-4-5")!;
	const payload = transformMessages(convertToLlm(messages), model);
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

	/**
	 * A session whose model answers from `replies`, repeating the last one. Each answer waits
	 * for `gate` when one is given, so a test can look at a run while it is under way.
	 */
	async function createSession(
		replies: AssistantMessage[] = [answer("all handled")],
		gate: Promise<void> = Promise.resolve(),
	): Promise<AgentSession> {
		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model, systemPrompt: "Test", tools: [] },
			streamFn: () => {
				const stream = new MockAssistantStream();
				const message = replies[Math.min(modelCalls, replies.length - 1)];
				modelCalls++;
				void gate.then(() => {
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

	it("settles the calls of an interrupted turn without calling the model", async () => {
		await createSession();
		seed([user("do two things"), assistant([call("hang-1"), call("hang-2")], "toolUse")]);

		expect(session.prepareStep()).toBe(true);
		expect(modelCalls).toBe(0);

		const messages = session.agent.state.messages;
		// The prompt was not added again, and both calls have a result a model can read.
		expect(messages.filter((m) => m.role === "user").length).toBe(1);
		const settled = messages.filter((m) => m.role === "toolResult") as ToolResultMessage[];
		expect(settled.map((m) => m.toolCallId)).toEqual(["hang-1", "hang-2"]);
		expect(settled.every((m) => isUnknownOutcome(m))).toBe(true);
		expect(findDanglingToolCalls(messages)).toEqual([]);

		// The settled results reached the session file, not just memory.
		const written = persisted().filter((m) => m.role === "toolResult") as ToolResultMessage[];
		expect(written.map((m) => m.toolCallId)).toEqual(["hang-1", "hang-2"]);
		assertValidToolPairing(messages);
	});

	it("leaves calls a custom message closed to the provider", async () => {
		await createSession();
		seed([user("go"), assistant([call("hang-1")], "toolUse")]);
		session.agent.state.messages = [...session.agent.state.messages, note("look at this")];

		// The provider reads the custom message as a user turn, which already answers the call.
		// A result recorded after it would pair with nothing.
		expect(session.prepareStep()).toBe(true);
		expect(session.agent.state.messages.filter((m) => m.role === "toolResult")).toEqual([]);
		assertValidToolPairing(session.agent.state.messages);
	});

	it("has work when the turn ends on a custom message", async () => {
		await createSession();
		seed([user("hi"), answer("done")]);
		session.agent.state.messages = [...session.agent.state.messages, note("one more thing")];

		expect(session.prepareStep()).toBe(true);
		expect(modelCalls).toBe(0);
	});

	it("records a prompt without calling the model", async () => {
		await createSession();

		expect(await session.recordPrompt("go")).toBe(true);

		expect(modelCalls).toBe(0);
		// The same entries prompt() writes, the loadout's system message included, so the turn is
		// on disk for whatever runs it, in this process or another one.
		expect(persisted().map((m) => m.role)).toEqual(["system", "user"]);
		expect(session.prepareStep()).toBe(true);
	});

	it("records the same entries prompt() writes ahead of the answer", async () => {
		const withoutTimes = (messages: AgentMessage[]) =>
			JSON.parse(JSON.stringify(messages, (key, value) => (key === "timestamp" ? undefined : value)));
		await createSession([answer("answer")]);
		await session.prompt("go");
		const prompted = persisted();
		expect(prompted.map((m) => m.role)).toEqual(["system", "user", "assistant"]);

		await createSession();
		expect(await session.recordPrompt("go")).toBe(true);

		expect(withoutTimes(persisted())).toEqual(withoutTimes(prompted.slice(0, -1)));
	});

	it("says it is busy while a run is under way", async () => {
		let release = () => {};
		await createSession(
			[answer("answer")],
			new Promise((resolve) => {
				release = resolve;
			}),
		);

		const running = session.prompt("go");
		await vi.waitFor(() => expect(session.isStreaming).toBe(true));
		// Data a driver can act on: "try again", not "the turn already has its answer".
		expect(session.prepareStep()).toBe("busy");

		release();
		await running;
		expect(session.prepareStep()).toBe(false);
	});

	it("drops an unanswered response from what the session file rebuilds, not only from memory", async () => {
		await createSession();
		seed([user("go"), assistant([{ type: "text", text: "" }], "aborted")]);

		expect(session.prepareStep()).toBe(true);

		// Memory is rebuilt from the file at every boundary, so the drop has to be there too.
		const rebuilt = sessionManager.buildSessionProjection().messages;
		expect(rebuilt.map((m) => m.role)).toEqual(["user"]);
		expect(session.agent.state.messages.map((m) => m.role)).toEqual(["user"]);
	});

	it("leaves a finished turn alone", async () => {
		await createSession();
		seed([user("hi"), answer("done")]);

		expect(session.prepareStep()).toBe(false);
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

	it("settles a call that the results recorded after it left out", async () => {
		await createSession();
		seed([
			user("do two things"),
			assistant([call("done-1"), call("hang-1")], "toolUse"),
			toolResult("done-1", "did done-1", false),
		]);

		expect(session.prepareStep()).toBe(true);

		const results = persisted().filter((m) => m.role === "toolResult") as ToolResultMessage[];
		expect(results.map((m) => m.toolCallId)).toEqual(["done-1", "hang-1"]);
		expect(isUnknownOutcome(results[1])).toBe(true);
		expect(session.agent.state.messages.at(-1)).toEqual(results[1]);
		assertValidToolPairing(session.agent.state.messages);
	});

	it("settles the calls a system message after them would hide", async () => {
		await createSession();
		seed([user("go"), assistant([call("hang-1")], "toolUse")]);
		const update = { role: "system" as const, content: "", timestamp: Date.now() };
		session.agent.state.messages = [...session.agent.state.messages, update];
		sessionManager.appendMessage(update);

		expect(session.prepareStep()).toBe(true);

		expect(isUnknownOutcome(session.agent.state.messages.at(-1))).toBe(true);
		assertValidToolPairing(session.agent.state.messages);
	});

	it("keeps memory and the file in step when a settling write throws", async () => {
		await createSession();
		seed([user("go"), assistant([call("hang-1"), call("hang-2")], "toolUse")]);
		const append = sessionManager.appendMessage.bind(sessionManager);
		let refuse = true;
		sessionManager.appendMessage = (message) => {
			if (refuse && message.role === "toolResult" && message.toolCallId === "hang-2") {
				throw new Error("not ours to write");
			}
			return append(message);
		};

		expect(() => session.prepareStep()).toThrow("not ours to write");
		const inMemory = session.agent.state.messages.filter((m) => m.role === "toolResult");
		expect(inMemory.map((m) => m.toolCallId)).toEqual(["hang-1"]);

		refuse = false;
		expect(session.prepareStep()).toBe(true);
		const written = persisted().filter((m) => m.role === "toolResult") as ToolResultMessage[];
		expect(written.map((m) => m.toolCallId)).toEqual(["hang-1", "hang-2"]);
		assertValidToolPairing(session.agent.state.messages);
	});

	it("refuses to record a prompt behind a call with no result", async () => {
		await createSession();
		seed([user("go"), assistant([call("hang-1")], "toolUse")]);

		await expect(session.recordPrompt("next")).rejects.toThrow("Call prepareStep() first");
		expect(session.agent.state.messages.map((m) => m.role)).toEqual(["user", "assistant"]);

		expect(session.prepareStep()).toBe(true);
		expect(await session.recordPrompt("next")).toBe(true);
		assertValidToolPairing(session.agent.state.messages);
	});
});

describe("AgentSession: recording a prompt", () => {
	it("offers a recorded prompt to the extensions, and records what they leave", async () => {
		const seen: string[] = [];
		const extension: ExtensionFactory = (pi) => {
			pi.on("message_start", (event) => {
				seen.push(`start:${event.message.role}`);
			});
			pi.on("message_end", (event) => {
				seen.push(`end:${event.message.role}`);
				if (event.message.role !== "user") return;
				const content = [{ type: "text" as const, text: "rewritten" }];
				return { message: { ...event.message, content } };
			});
		};
		const harness = await createHarness({ extensionFactories: [extension] });
		try {
			expect(await harness.session.recordPrompt("go")).toBe(true);

			const userEvents = seen.filter((entry) => entry.endsWith(":user"));
			expect(userEvents).toEqual(["start:user", "end:user"]);
			expect(getUserTexts(harness)).toEqual(["rewritten"]);
			const written = harness.sessionManager
				.getBranch()
				.flatMap((entry) => (entry.type === "message" ? [entry.message] : []))
				.filter((message) => message.role === "user");
			expect(written.map((message) => getMessageText(message))).toEqual(["rewritten"]);
		} finally {
			harness.cleanup();
		}
	});

	it("holds the session busy while a prompt is being recorded", async () => {
		const ref: { session?: AgentSession; seen?: boolean | "busy" } = {};
		const extension: ExtensionFactory = (pi) => {
			pi.on("message_end", (event) => {
				if (event.message.role === "user") ref.seen ??= ref.session?.prepareStep();
			});
		};
		const harness = await createHarness({ extensionFactories: [extension] });
		try {
			ref.session = harness.session;

			expect(await harness.session.recordPrompt("go")).toBe(true);

			expect(ref.seen).toBe("busy");
			expect(harness.session.isIdle).toBe(true);
			expect(harness.session.prepareStep()).toBe(true);
		} finally {
			harness.cleanup();
		}
	});

	it("records a message sent while the prompt was recorded after the prompt", async () => {
		const extension: ExtensionFactory = (pi) => {
			pi.on("message_end", (event) => {
				if (event.message.role !== "user") return;
				const note = { customType: "note", content: "noted", display: true };
				pi.sendMessage(note, { triggerTurn: false });
			});
		};
		const harness = await createHarness({ extensionFactories: [extension] });
		try {
			harness.setResponses([fauxAssistantMessage("answer")]);

			expect(await harness.session.recordPrompt("go")).toBe(true);

			const roles = harness.session.agent.state.messages.map((message) => message.role);
			expect(roles.slice(-2)).toEqual(["user", "custom"]);
		} finally {
			harness.cleanup();
		}
	});
});
