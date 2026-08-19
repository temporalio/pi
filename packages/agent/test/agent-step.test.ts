import {
	type AssistantMessage,
	type AssistantMessageEvent,
	EventStream,
	type Message,
	type Model,
	type UserMessage,
} from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import { agentStep } from "../src/agent-loop.ts";
import type { AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, AgentTool } from "../src/types.ts";

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

function createUsage() {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function createModel(): Model<"openai-responses"> {
	return {
		id: "mock",
		name: "mock",
		api: "openai-responses",
		provider: "openai",
		baseUrl: "https://example.invalid",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 8192,
		maxTokens: 2048,
	};
}

function createAssistantMessage(
	content: AssistantMessage["content"],
	stopReason: AssistantMessage["stopReason"] = "stop",
): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "openai-responses",
		provider: "openai",
		model: "mock",
		usage: createUsage(),
		stopReason,
		timestamp: Date.now(),
	};
}

function createUserMessage(text: string): UserMessage {
	return { role: "user", content: text, timestamp: Date.now() };
}

function identityConverter(messages: AgentMessage[]): Message[] {
	return messages.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult") as Message[];
}

describe("agentStep", () => {
	it("runs exactly one model call and its tools, then stops", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: string[] = [];
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_id, params) {
				executed.push(params.value);
				return { content: [{ type: "text", text: `echoed: ${params.value}` }], details: { value: params.value } };
			},
		};

		let callCount = 0;
		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				callCount++;
				if (callCount === 1) {
					stream.push({
						type: "done",
						reason: "toolUse",
						message: createAssistantMessage(
							[{ type: "toolCall", id: "t1", name: "echo", arguments: { value: "x" } }],
							"toolUse",
						),
					});
				} else {
					// A whole-turn loop would take this branch; a single step must not.
					stream.push({
						type: "done",
						reason: "stop",
						message: createAssistantMessage([{ type: "text", text: "more" }]),
					});
				}
			});
			return stream;
		};

		const context: AgentContext = { systemPrompt: "", messages: [createUserMessage("go")], tools: [tool] };
		const config: AgentLoopConfig = { model: createModel(), convertToLlm: identityConverter };

		const events: AgentEvent[] = [];
		const stream = agentStep(context, config, undefined, streamFn);
		for await (const event of stream) {
			events.push(event);
		}
		const outcome = await stream.result();

		// Exactly one model call, and the tool it requested ran once.
		expect(callCount).toBe(1);
		expect(executed).toEqual(["x"]);

		// One turn boundary. A step is a turn, so it opens no run of its own.
		expect(events.filter((e) => e.type === "turn_start").length).toBe(1);
		expect(events.filter((e) => e.type === "turn_end").length).toBe(1);
		expect(events.filter((e) => e.type === "agent_start").length).toBe(0);
		expect(events.filter((e) => e.type === "agent_end").length).toBe(0);

		// The tool result still needs an answer, so the caller has to step again.
		expect(outcome.hasMoreToolCalls).toBe(true);

		// The step produced the assistant message and its tool result, and nothing more.
		expect(outcome.messages.map((m) => m.role)).toEqual(["assistant", "toolResult"]);
	});

	it("throws when the last message is an assistant", () => {
		const context: AgentContext = {
			systemPrompt: "",
			messages: [createAssistantMessage([{ type: "text", text: "hi" }])],
			tools: [],
		};
		expect(() =>
			agentStep(
				context,
				{ model: createModel(), convertToLlm: identityConverter },
				undefined,
				() => new MockAssistantStream(),
			),
		).toThrow(/Cannot step from message role: assistant/);
	});
});
