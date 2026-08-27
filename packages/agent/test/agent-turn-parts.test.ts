// The three parts of a step, driven directly. What is pinned here is the contract a caller
// outside the loop depends on: the model call records the calls and runs none of them, a call
// reports its result rather than entering it, and the seal enters them in the model's order.
// The last test is the one that matters most: a turn split into three parts and a turn the
// loop runs itself have to leave the same transcript, or the two drivers have drifted.
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
import {
	runAgentModelCall,
	runAgentSeal,
	runAgentStep,
	runAgentToolCall,
	type TurnToolCallOutcome,
} from "../src/agent-loop.ts";
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

const LLM_ROLES = ["user", "assistant", "toolResult"];

function identityConverter(messages: AgentMessage[]): Message[] {
	return messages.filter((m) => LLM_ROLES.includes(m.role)) as Message[];
}

const schema = Type.Object({ value: Type.String() });

interface ProbeTool {
	tool: AgentTool<typeof schema, { value: string }>;
	ran: string[];
}

function probeTool(options: { name?: string; terminate?: boolean; sequential?: boolean } = {}): ProbeTool {
	const ran: string[] = [];
	const tool: AgentTool<typeof schema, { value: string }> = {
		name: options.name ?? "probe",
		label: "Probe",
		description: "Probe tool",
		parameters: schema,
		executionMode: options.sequential ? "sequential" : undefined,
		async execute(_id, params) {
			ran.push(params.value);
			return {
				content: [{ type: "text", text: `ran: ${params.value}` }],
				details: { value: params.value },
				...(options.terminate ? { terminate: true } : {}),
			};
		},
	};
	return { tool, ran };
}

function toolCall(id: string, name = "probe", value = id) {
	return { type: "toolCall", id, name, arguments: { value } } as const;
}

/** A model that answers with the given responses in order, and counts what it was asked. */
function scriptedModel(responses: Array<() => AssistantMessage>) {
	const asked = { count: 0 };
	const streamFn = () => {
		const stream = new MockAssistantStream();
		queueMicrotask(() => {
			const next = responses[Math.min(asked.count, responses.length - 1)];
			asked.count++;
			// The loop reads the message's own stop reason, so the event's is only a label.
			stream.push({ type: "done", reason: "stop", message: next() });
		});
		return stream;
	};
	return { streamFn, asked };
}

function collector() {
	const events: AgentEvent[] = [];
	return { events, emit: async (event: AgentEvent) => void events.push(event) };
}

function contextWith(messages: AgentMessage[], tools: AgentContext["tools"]): AgentContext {
	return { systemPrompt: "", messages, tools };
}

const config = (): AgentLoopConfig => ({ model: createModel(), convertToLlm: identityConverter });

describe("model call", () => {
	it("records the calls it was asked for and runs none of them", async () => {
		const { tool, ran } = probeTool();
		const { streamFn, asked } = scriptedModel([
			() => createAssistantMessage([toolCall("t1"), toolCall("t2")], "toolUse"),
		]);
		const context = contextWith([createUserMessage("go")], [tool]);
		const { emit, events } = collector();

		const outcome = await runAgentModelCall(context, config(), emit, undefined, streamFn);

		expect(asked.count).toBe(1);
		expect(outcome.toolCalls.map((c) => c.id)).toEqual(["t1", "t2"]);
		expect(outcome.ended).toBe(false);
		expect(outcome.replayed).toBe(false);
		// Nothing ran, and the transcript ends on the message that asked, so the calls are
		// there for whoever runs them.
		expect(ran).toEqual([]);
		expect(context.messages.map((m) => m.role)).toEqual(["user", "assistant"]);
		expect(events.filter((e) => e.type === "tool_execution_start")).toHaveLength(0);
	});

	it("does not ask the model twice for a response the transcript already holds", async () => {
		const { tool } = probeTool();
		const { streamFn, asked } = scriptedModel([() => createAssistantMessage([toolCall("t9")], "toolUse")]);
		const context = contextWith(
			[createUserMessage("go"), createAssistantMessage([toolCall("t1")], "toolUse")],
			[tool],
		);
		const { emit } = collector();

		const outcome = await runAgentModelCall(context, config(), emit, undefined, streamFn);

		// A driver whose record of the call was lost gets the recorded calls back. Asking again
		// would pay for a second response and answer calls that nothing has run.
		expect(asked.count).toBe(0);
		expect(outcome.replayed).toBe(true);
		expect(outcome.toolCalls.map((c) => c.id)).toEqual(["t1"]);
	});

	it("says when the calls have to run one at a time", async () => {
		const ordered = probeTool({ name: "ordered", sequential: true });
		const free = probeTool({ name: "free" });
		const { streamFn } = scriptedModel([
			() => createAssistantMessage([toolCall("t1", "free"), toolCall("t2", "ordered")], "toolUse"),
		]);
		const context = contextWith([createUserMessage("go")], [free.tool, ordered.tool]);
		const { emit } = collector();

		const outcome = await runAgentModelCall(context, config(), emit, undefined, streamFn);

		// One sequential tool makes the whole batch sequential, and a caller that fans the calls
		// out has no other way to know.
		expect(outcome.sequential).toBe(true);
	});

	it("reports a response that ended the run, and the seal leaves it alone", async () => {
		const { tool } = probeTool();
		const { streamFn } = scriptedModel([() => createAssistantMessage([{ type: "text", text: "" }], "error")]);
		const context = contextWith([createUserMessage("go")], [tool]);
		const { emit, events } = collector();

		const outcome = await runAgentModelCall(context, config(), emit, undefined, streamFn);
		expect(outcome.ended).toBe(true);
		expect(events.filter((e) => e.type === "agent_end")).toHaveLength(1);

		const sealed = await runAgentSeal(context, config(), [], emit);

		// The model call closed the turn as it went. A second turn_end would tell the transcript
		// the turn ended twice.
		expect(sealed.hasMoreToolCalls).toBe(false);
		expect(events.filter((e) => e.type === "turn_end")).toHaveLength(1);
	});
});

describe("tool call", () => {
	it("reports its result instead of entering it in the transcript", async () => {
		const { tool, ran } = probeTool();
		const context = contextWith(
			[createUserMessage("go"), createAssistantMessage([toolCall("t1")], "toolUse")],
			[tool],
		);
		const { emit, events } = collector();

		const outcome = await runAgentToolCall(context, config(), "t1", emit, undefined);

		expect(ran).toEqual(["t1"]);
		expect(outcome?.message.toolCallId).toBe("t1");
		// The results of a step go in together, at the seal, so the transcript is untouched here.
		expect(context.messages.map((m) => m.role)).toEqual(["user", "assistant"]);
		expect(events.filter((e) => e.type === "message_end")).toHaveLength(0);
		expect(events.filter((e) => e.type === "tool_execution_end")).toHaveLength(1);
	});

	it("does not run a call the transcript already answered", async () => {
		const { tool, ran } = probeTool();
		const context = contextWith(
			[
				createUserMessage("go"),
				createAssistantMessage([toolCall("t1")], "toolUse"),
				{
					role: "toolResult",
					toolCallId: "t1",
					toolName: "probe",
					content: [{ type: "text", text: "ran: t1" }],
					details: {},
					isError: false,
					timestamp: Date.now(),
				},
			],
			[tool],
		);
		const { emit } = collector();

		const outcome = await runAgentToolCall(context, config(), "t1", emit, undefined);

		// The at-least-once case. Re-running a tool that already had its effect is the worse
		// failure, so a settled call is left alone.
		expect(outcome).toBeUndefined();
		expect(ran).toEqual([]);
	});

	it("refuses a call the current step never recorded", async () => {
		const { tool } = probeTool();
		const context = contextWith(
			[createUserMessage("go"), createAssistantMessage([toolCall("t1")], "toolUse")],
			[tool],
		);
		const { emit } = collector();

		await expect(runAgentToolCall(context, config(), "t404", emit, undefined)).rejects.toThrow(
			/No recorded tool call t404/,
		);
	});
});

describe("seal", () => {
	const recordedFor = (context: AgentContext) =>
		context.messages.filter((m) => m.role === "toolResult").map((m) => m.toolCallId);

	const stepWithTwoCalls = async () => {
		const { tool, ran } = probeTool();
		const { streamFn } = scriptedModel([() => createAssistantMessage([toolCall("t1"), toolCall("t2")], "toolUse")]);
		const context = contextWith([createUserMessage("go")], [tool]);
		const { emit, events } = collector();
		const model = await runAgentModelCall(context, config(), emit, undefined, streamFn);
		return { context, emit, events, model, ran };
	};

	it("records the results in the order the model asked, not the order they settled", async () => {
		const { context, emit, model } = await stepWithTwoCalls();

		// Settle the second call first, the way concurrent dispatch does.
		const second = await runAgentToolCall(context, config(), model.toolCalls[1].id, emit, undefined);
		const first = await runAgentToolCall(context, config(), model.toolCalls[0].id, emit, undefined);

		await runAgentSeal(context, config(), [first, second] as TurnToolCallOutcome[], emit);

		expect(recordedFor(context)).toEqual(["t1", "t2"]);
	});

	it("records each result once when it runs twice", async () => {
		const { context, emit, model } = await stepWithTwoCalls();
		const results = [
			await runAgentToolCall(context, config(), model.toolCalls[0].id, emit, undefined),
			await runAgentToolCall(context, config(), model.toolCalls[1].id, emit, undefined),
		] as TurnToolCallOutcome[];

		const once = await runAgentSeal(context, config(), results, emit);
		const twice = await runAgentSeal(context, config(), results, emit);

		// A seal whose answer never reached its caller runs again. Recording the results a
		// second time would show the model its own tools twice.
		expect(recordedFor(context)).toEqual(["t1", "t2"]);
		// And the turn still has to carry on, which counting only the new results would lose.
		expect(once.hasMoreToolCalls).toBe(true);
		expect(twice.hasMoreToolCalls).toBe(true);
	});

	it("ends the turn only when every call asked it to", async () => {
		const stopping = probeTool({ name: "stopping", terminate: true });
		const plain = probeTool({ name: "plain" });
		const { streamFn } = scriptedModel([
			() => createAssistantMessage([toolCall("t1", "stopping"), toolCall("t2", "plain")], "toolUse"),
		]);
		const context = contextWith([createUserMessage("go")], [stopping.tool, plain.tool]);
		const { emit } = collector();
		const model = await runAgentModelCall(context, config(), emit, undefined, streamFn);
		const results = [
			await runAgentToolCall(context, config(), model.toolCalls[0].id, emit, undefined),
			await runAgentToolCall(context, config(), model.toolCalls[1].id, emit, undefined),
		] as TurnToolCallOutcome[];

		const mixed = await runAgentSeal(context, config(), results, emit);
		expect(mixed.hasMoreToolCalls).toBe(true);

		const stoppingOnly = await runAgentSeal(context, config(), [results[0]], emit);
		expect(stoppingOnly.hasMoreToolCalls).toBe(false);
	});
});

describe("the split turn and the whole turn", () => {
	// Two tools and two calls, one of them a tool the model was never given, so the step also
	// carries a failure through both drivers.
	const script = () =>
		scriptedModel([() => createAssistantMessage([toolCall("t1"), toolCall("t2", "missing")], "toolUse")]);

	const transcript = (messages: AgentMessage[]) =>
		messages.map((m) => ({
			role: m.role,
			...(m.role === "toolResult" ? { call: m.toolCallId, error: m.isError } : {}),
		}));

	it("leave the same transcript", async () => {
		const whole = probeTool();
		const wholeContext = contextWith([createUserMessage("go")], [whole.tool]);
		const wholeEvents = collector();
		const wholeOutcome = await runAgentStep(wholeContext, config(), wholeEvents.emit, undefined, script().streamFn);

		const split = probeTool();
		const splitContext = contextWith([createUserMessage("go")], [split.tool]);
		const splitEvents = collector();
		const model = await runAgentModelCall(splitContext, config(), splitEvents.emit, undefined, script().streamFn);
		const results: TurnToolCallOutcome[] = [];
		for (const call of model.toolCalls) {
			const result = await runAgentToolCall(splitContext, config(), call.id, splitEvents.emit, undefined);
			if (result) results.push(result);
		}
		const splitOutcome = await runAgentSeal(splitContext, config(), results, splitEvents.emit);

		// Spelled out as well as compared, so two drivers that both produced nothing would not
		// agree their way past this.
		expect(transcript(wholeContext.messages)).toEqual([
			{ role: "user" },
			{ role: "assistant" },
			{ role: "toolResult", call: "t1", error: false },
			{ role: "toolResult", call: "t2", error: true },
		]);
		expect(transcript(splitContext.messages)).toEqual(transcript(wholeContext.messages));
		expect(split.ran).toEqual(whole.ran);
		expect(splitOutcome.hasMoreToolCalls).toBe(wholeOutcome.hasMoreToolCalls);
		// The turn boundary is one turn either way. A split that opened a turn per part would
		// show up here before it showed up in a session file.
		const turnEvents = (events: AgentEvent[]) => events.filter((e) => e.type.startsWith("turn_")).map((e) => e.type);
		expect(turnEvents(wholeEvents.events)).toEqual(["turn_start", "turn_end"]);
		expect(turnEvents(splitEvents.events)).toEqual(turnEvents(wholeEvents.events));
	});
});
