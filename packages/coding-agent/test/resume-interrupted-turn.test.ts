import { existsSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, type AgentMessage, type AgentTool } from "@earendil-works/pi-agent-core";
import { transformMessages } from "@earendil-works/pi-ai/api/transform-messages";
import {
	type AssistantMessage,
	type AssistantMessageEvent,
	contentText,
	EventStream,
	fauxAssistantMessage,
	getCurrentSystemPrompt,
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
		tools?: Record<string, AgentTool>,
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
			baseToolsOverride: tools,
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

		await session.sendCustomMessage(
			{ customType: "note", content: "context", display: false },
			{
				deliverAs: "nextTurn",
			},
		);

		await expect(session.recordPrompt("next")).rejects.toThrow("Call prepareStep() first");
		expect(session.agent.state.messages.map((m) => m.role)).toEqual(["user", "assistant"]);

		expect(session.prepareStep()).toBe(true);
		expect(await session.recordPrompt("next")).toBe(true);
		assertValidToolPairing(session.agent.state.messages);
		// The refusal took nothing, so the queued message still goes in with the prompt.
		const notes = session.agent.state.messages.filter((m) => m.role === "custom" && m.customType === "note");
		expect(notes).toHaveLength(1);
	});

	it("queues a recording sent mid-run instead of rejecting it", async () => {
		let release = () => {};
		await createSession(
			[answer("first"), answer("second")],
			new Promise((resolve) => {
				release = resolve;
			}),
		);

		const running = session.prompt("go");
		await vi.waitFor(() => expect(session.isStreaming).toBe(true));
		// The run owns the transcript: the recording queues behind it, as prompt() would.
		await expect(session.recordPrompt("new direction", { streamingBehavior: "followUp" })).resolves.toBe(false);
		expect(session.prepareStep()).toBe("busy");

		release();
		await running;
		const texts = session.agent.state.messages
			.filter((m) => m.role === "user")
			.map((m) => contentText(m.content, ""));
		expect(texts).toContain("new direction");
	});

	it("runs the winner of two concurrent prompts with its own system prompt", async () => {
		const extension: ExtensionFactory = (pi) => {
			pi.on("before_agent_start", (event) => ({ systemPrompt: `for ${event.prompt}` }));
		};
		const harness = await createHarness({ extensionFactories: [extension] });
		try {
			const seen: string[] = [];
			harness.setResponses([
				(context) => {
					seen.push(getCurrentSystemPrompt(context.messages) ?? "");
					return fauxAssistantMessage("answer");
				},
			]);

			// The loser's build must not put its prompt options under the winner's turn.
			const results = await Promise.allSettled([harness.session.prompt("one"), harness.session.prompt("two")]);
			expect(results.filter((result) => result.status === "rejected")).toHaveLength(1);
			const winner = results[0].status === "fulfilled" ? "one" : "two";
			expect(seen).toEqual([`for ${winner}`]);
		} finally {
			harness.cleanup();
		}
	});

	it("refuses a prompt when a branch summary started while it was built", async () => {
		let release = () => {};
		const held = new Promise<void>((resolve) => {
			release = resolve;
		});
		const extension: ExtensionFactory = (pi) => {
			pi.on("before_agent_start", async () => {
				await held;
			});
		};
		const harness = await createHarness({ extensionFactories: [extension] });
		try {
			harness.setResponses([fauxAssistantMessage("answer")]);
			const running = harness.session.prompt("go");
			await new Promise((resolve) => setTimeout(resolve, 10));

			// It started after the build's first check, and it rewrites the transcript too.
			const open = harness.session as unknown as { _branchSummaryAbortController?: AbortController };
			open._branchSummaryAbortController = new AbortController();
			release();
			await expect(running).rejects.toThrow("compaction is in progress");
			open._branchSummaryAbortController = undefined;

			expect(harness.session.isIdle).toBe(true);
			expect(harness.getPendingResponseCount()).toBe(1);
		} finally {
			harness.cleanup();
		}
	});

	it("gives the session back when a turn fails to start", async () => {
		await createSession();
		const internals = session as unknown as { _beginTurn: () => void };
		internals._beginTurn = () => {
			throw new Error("could not record the selection");
		};

		await expect(session.prompt("go")).rejects.toThrow("could not record the selection");
		expect(session.isIdle).toBe(true);
		expect(modelCalls).toBe(0);
	});

	it("tells the loser of two concurrent starts that the session is taken", async () => {
		let release = () => {};
		await createSession(
			[answer("answer")],
			new Promise((resolve) => {
				release = resolve;
			}),
		);

		// Both builders pass their awaits before either caller claims the run.
		const record = session.recordPrompt("checkpoint");
		const run = session.prompt("go");
		release();
		const results = await Promise.allSettled([record, run]);

		const rejected = results.filter((r) => r.status === "rejected") as PromiseRejectedResult[];
		expect(rejected.length).toBe(1);
		expect(String(rejected[0].reason)).toContain("already processing");

		// Exactly one of the two reached the transcript.
		const texts = session.agent.state.messages
			.filter((m) => m.role === "user")
			.map((m) => contentText(m.content, ""));
		expect(texts.includes("checkpoint") !== texts.includes("go")).toBe(true);
	});

	it("settles once when a subscriber calls back in from message_end", async () => {
		await createSession();
		seed([user("go"), assistant([call("hang-1"), call("hang-2")], "toolUse")]);

		const nested: (boolean | "busy")[] = [];
		const unsubscribe = session.subscribe((event) => {
			if (event.type === "message_end") nested.push(session.prepareStep());
		});
		expect(session.prepareStep()).toBe(true);
		unsubscribe();

		expect(nested).toContain("busy");
		const results = session.agent.state.messages.filter((m) => m.role === "toolResult");
		expect(results.map((m) => m.toolCallId)).toEqual(["hang-1", "hang-2"]);
		assertValidToolPairing(session.agent.state.messages);
	});

	it("queues a turn a subscriber starts while a stopped turn settles", async () => {
		await createSession();
		seed([user("go"), assistant([call("hang-1"), call("hang-2")], "toolUse")]);
		const unsubscribe = session.subscribe((event) => {
			if (event.type !== "message_end" || event.message.role !== "toolResult") return;
			if (event.message.toolCallId !== "hang-1") return;
			void session.sendCustomMessage(
				{ customType: "nudge", content: "go on", display: false },
				{ triggerTurn: true },
			);
		});
		expect(session.prepareStep()).toBe(true);
		unsubscribe();
		await new Promise((resolve) => setTimeout(resolve, 20));

		// No turn started from the part settled when the first result was announced.
		expect(modelCalls).toBe(0);
		const results = session.agent.state.messages.filter((m) => m.role === "toolResult");
		expect(results.map((m) => m.toolCallId)).toEqual(["hang-1", "hang-2"]);
		expect(session.agent.hasQueuedMessages()).toBe(true);
	});

	it("queues a turn a subscriber starts while a recording flushes what it queued", async () => {
		await createSession([answer("answer")]);
		const unsubscribe = session.subscribe((event) => {
			if (event.type !== "message_end") return;
			const message = event.message;
			if (message.role === "user") {
				// Held for the end of the busy window, then flushed.
				void session.sendCustomMessage(
					{ customType: "note", content: "first", display: false },
					{ triggerTurn: false },
				);
			} else if (message.role === "custom" && message.customType === "note") {
				void session.sendCustomMessage(
					{ customType: "nudge", content: "go on", display: false },
					{ triggerTurn: true },
				);
			}
		});
		expect(await session.recordPrompt("go")).toBe(true);
		unsubscribe();
		await new Promise((resolve) => setTimeout(resolve, 20));

		expect(modelCalls).toBe(0);
		expect(session.agent.hasQueuedMessages()).toBe(true);
	});

	it("reads a turn its tools stopped as answered, also after a reopen", async () => {
		const stop = {
			name: "do",
			label: "Do",
			description: "Asks the turn to stop",
			parameters: { type: "object", properties: {} },
			execute: async () => ({ content: [{ type: "text", text: "stopped" }], details: {}, terminate: true }),
		} as unknown as AgentTool;
		await createSession([assistant([call("stop-1")], "toolUse"), answer("never")], undefined, { do: stop });

		await session.prompt("go");
		expect(modelCalls).toBe(1);
		expect(session.agent.state.messages.at(-1)?.role).toBe("toolResult");
		expect(session.prepareStep()).toBe(false);

		// A driver that opens the file again must not ask the model to answer a stopped turn.
		const reopened = SessionManager.open(sessionManager.getSessionFile()!);
		const projection = reopened.buildSessionProjection().messages;
		session.agent.state.messages = projection;
		expect(session.prepareStep()).toBe(false);
	});

	it("is busy while compaction runs", async () => {
		await createSession();
		const open = session as unknown as { _compactionAbortController?: AbortController };
		open._compactionAbortController = new AbortController();
		expect(session.prepareStep()).toBe("busy");
		open._compactionAbortController = undefined;
		expect(session.prepareStep()).toBe(false);
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
