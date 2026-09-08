// Preparation belongs to the turn that follows the one that just ended, and the loop keeps the
// turn it ended in a local. A caller stepping from outside has no such local, so every external
// entry point missed it: `prompt()` prepared and `step()` did not, on the same agent, with the
// same responses. A review reproduced that with a callback that changes the system prompt.
//
// What is checked here is the whole contract, not the callback firing: preparation runs once per
// completed step, what it returns reaches the calls and the seal that follow, a replayed model
// response neither prepares nor asks the provider again, and a retried model call reuses the
// preparation rather than running it a second time.

import { type AssistantMessage, type AssistantMessageEvent, EventStream } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import { Agent } from "../src/agent.ts";
import { runAgentModelCall, type StepCursor } from "../src/agent-loop.ts";
import type { AgentContext, AgentLoopConfig, AgentMessage, AgentTool, PrepareNextTurnContext } from "../src/types.ts";

const LLM_ROLES = ["user", "assistant", "toolResult"];

const echoSchema = Type.Object({ value: Type.String() });
const echoTool = {
	name: "echo",
	label: "Echo",
	description: "Echo tool",
	parameters: echoSchema,
	execute: async (args: { value: string }) => ({ output: args.value }),
} as unknown as AgentTool;

class Response extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor() {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") return event.message;
				if (event.type === "error") return event.error;
				throw new Error("not a final response");
			},
		);
	}
}

interface Recorded {
	prepared: number;
	prompts: string[];
	models: Array<string | undefined>;
	/** The system prompt each phase after the model call saw. */
	toolPhase: string[];
	sealPhase: string[];
}

function agentUnderTest(options: { failCall?: number } = {}) {
	const recorded: Recorded = { prepared: 0, prompts: [], models: [], toolPhase: [], sealPhase: [] };
	let calls = 0;
	const agent = new Agent({
		initialState: { systemPrompt: "original", tools: [echoTool] },
		beforeToolCall: async (context) => {
			recorded.toolPhase.push(context.context.systemPrompt ?? "");
			return undefined;
		},
		shouldStopAfterTurn: (context) => {
			recorded.sealPhase.push(context.context.systemPrompt ?? "");
			return false;
		},
		prepareNextTurnWithContext: async (turn) => {
			recorded.prepared++;
			// The genuine completed turn, not something rebuilt from the transcript.
			expect(turn.toolResults.length).toBe(1);
			expect(turn.message.content.some((block) => block.type === "toolCall")).toBe(true);
			return { context: { ...turn.context, systemPrompt: "prepared" }, model: "prepared-model" as never };
		},
		streamFn: (model, context) => {
			calls++;
			recorded.prompts.push(context.systemPrompt ?? "");
			recorded.models.push(model as unknown as string);
			const stream = new Response();
			if (options.failCall === calls) throw new Error("provider is down");
			// The first two responses ask for a tool, so the step after preparation has a tool phase
			// and a seal to observe. The third ends the turn.
			const first = recorded.prompts.length <= 2;
			const message: AssistantMessage = {
				role: "assistant",
				content: first
					? [{ type: "toolCall", id: `call-${calls}`, name: "echo", arguments: { value: "x" } }]
					: [{ type: "text", text: "done" }],
				api: "openai-responses",
				provider: "openai",
				model: "mock",
				timestamp: Date.now(),
				stopReason: first ? "toolUse" : "stop",
				usage: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
			};
			queueMicrotask(() => stream.push({ type: "done", reason: first ? "toolUse" : "stop", message }));
			return stream;
		},
	});
	agent.state.messages = [{ role: "user", content: "run", timestamp: Date.now() }];
	return { agent, recorded };
}

describe("the step cursor", () => {
	it("prepares between two whole steps, as the loop does between two turns", async () => {
		const { agent, recorded } = agentUnderTest();
		await agent.step();
		await agent.step();

		expect(recorded.prepared).toBe(1);
		expect(recorded.prompts).toEqual(["original", "prepared"]);
	});

	it("carries what preparation returned into the calls and the seal of that step", async () => {
		const { agent, recorded } = agentUnderTest();
		// Step one prepares nothing and its tool runs under the original prompt.
		await agent.step();
		expect(recorded.toolPhase).toEqual(["original"]);
		expect(recorded.sealPhase).toEqual(["original"]);

		// Step two prepares. Everything that step does afterwards belongs to what it prepared, not
		// to the agent's state, which still says `original`.
		const model = await agent.modelCall();
		expect(recorded.prepared).toBe(1);
		expect(recorded.prompts).toEqual(["original", "prepared"]);
		expect(recorded.models[1]).toBe("prepared-model");
		expect(model.replayed).toBe(false);
		expect(agent.state.systemPrompt).toBe("original");

		const result = await agent.runToolCall(model.toolCalls[0].id);
		await agent.sealStep(result ? [result] : []);
		expect(recorded.toolPhase).toEqual(["original", "prepared"]);
		expect(recorded.sealPhase).toEqual(["original", "prepared"]);
	});

	// A replay is the same step arriving again, so it must leave the completed turn where it is:
	// the model call it replays either already prepared or has still to.
	it("leaves the completed turn alone when a response is replayed", async () => {
		let prepared = 0;
		const cursor: StepCursor = {
			previousTurn: {
				message: { role: "assistant", content: [{ type: "text", text: "done" }] },
				toolResults: [],
				context: { systemPrompt: "original", messages: [], tools: [] },
				newMessages: [],
			} as unknown as PrepareNextTurnContext,
		};
		const context: AgentContext = {
			systemPrompt: "original",
			tools: [],
			messages: [
				{ role: "user", content: "run", timestamp: Date.now() },
				{
					role: "assistant",
					content: [{ type: "toolCall", id: "call", name: "echo", arguments: {} }],
					stopReason: "toolUse",
					timestamp: Date.now(),
				} as unknown as AgentMessage,
			],
		};
		const config = {
			convertToLlm: (messages: AgentMessage[]) => messages.filter((m) => LLM_ROLES.includes(m.role)),
			prepareNextTurn: async () => {
				prepared++;
				return undefined;
			},
		} as unknown as AgentLoopConfig;

		const outcome = await runAgentModelCall(context, config, async () => {}, undefined, undefined as never, cursor);

		expect(outcome.replayed).toBe(true);
		expect(prepared).toBe(0);
		expect(cursor.previousTurn).toBeDefined();
	});

	it("does not prepare or call the provider again for a replayed response", async () => {
		const { agent, recorded } = agentUnderTest();
		const first = await agent.modelCall();
		const replay = await agent.modelCall();

		expect(replay.replayed).toBe(true);
		expect(recorded.prompts.length).toBe(1);
		expect(recorded.prepared).toBe(0);
		expect(replay.toolCalls.map((call) => call.id)).toEqual(first.toolCalls.map((call) => call.id));
	});

	// An attempt that dies inside the provider has still prepared: the app's callback ran, and that
	// callback is where an app compacts. The attempt that replaces it is the same step again, so it
	// must reuse that preparation. Driven at the loop entry point, because a failed attempt has to
	// be settled before `Agent.modelCall()` will start another one, and settling is the driver's.
	it("prepares once across two attempts at the same step", async () => {
		let prepared = 0;
		const prompts: string[] = [];
		const cursor: StepCursor = {
			previousTurn: {
				message: { role: "assistant", content: [{ type: "text", text: "done" }] },
				toolResults: [],
				context: { systemPrompt: "original", messages: [], tools: [] },
				newMessages: [],
			} as unknown as PrepareNextTurnContext,
		};
		const context: AgentContext = {
			systemPrompt: "original",
			messages: [{ role: "user", content: "run", timestamp: Date.now() }],
			tools: [],
		};
		const config = {
			convertToLlm: (messages: AgentMessage[]) => messages.filter((m) => LLM_ROLES.includes(m.role)),
			prepareNextTurn: async () => {
				prepared++;
				return { context: { ...context, systemPrompt: "prepared" } };
			},
		} as unknown as AgentLoopConfig;
		const streamFn = ((_model: unknown, ctx: AgentContext) => {
			prompts.push(ctx.systemPrompt ?? "");
			throw new Error("provider is down");
		}) as never;

		for (const _attempt of [1, 2]) {
			await expect(runAgentModelCall(context, config, async () => {}, undefined, streamFn, cursor)).rejects.toThrow(
				"provider is down",
			);
		}

		expect(prepared).toBe(1);
		expect(prompts).toEqual(["prepared", "prepared"]);
		expect(cursor.previousTurn).toBeUndefined();
	});

	it("starts a prompt from no completed turn, whatever a stepping caller left behind", async () => {
		const { agent, recorded } = agentUnderTest();
		await agent.step();
		expect(recorded.prepared).toBe(0);

		// A new run is not a continuation of the step before it: the loop begins with no completed
		// turn, so preparation must not fire on the run's own first model call.
		agent.state.messages = [{ role: "user", content: "again", timestamp: Date.now() }];
		await agent.prompt("again");

		expect(recorded.prompts[1]).toBe("original");
	});
});
