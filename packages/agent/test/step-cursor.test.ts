// Preparation belongs to the turn that follows the one that just ended, and the loop keeps the
// turn it ended in a local. A caller stepping from outside has no such local, so the agent holds
// it for them.
//
// What is checked here is the whole contract, not the callback firing: preparation runs once per
// completed step, what it returns reaches the calls and the seal that follow, a replayed model
// response neither prepares nor asks the provider again, and a retried model call reuses the
// preparation rather than running it a second time.

import { type AssistantMessage, createAssistantMessageEventStream } from "@earendil-works/pi-ai";
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

/** Which context a phase ran against. Preparation marks the one it returns. */
type Labelled = AgentContext & { label?: string };
const labelOf = (context: AgentContext) => (context as Labelled).label ?? "original";
/** The provider sees no context fields, only the model, so preparation marks the model too. */
const PREPARED_MODEL = "prepared-model";
const promptOf = (model: unknown) => (model === PREPARED_MODEL ? "prepared" : "original");

interface Recorded {
	prepared: number;
	prompts: string[];
	models: Array<string | undefined>;
	/** The context each phase after the model call saw. */
	toolPhase: string[];
	sealPhase: string[];
}

function agentUnderTest(options: { prepareOnce?: boolean } = {}) {
	const recorded: Recorded = { prepared: 0, prompts: [], models: [], toolPhase: [], sealPhase: [] };
	let calls = 0;
	const agent = new Agent({
		initialState: { tools: [echoTool] },
		beforeToolCall: async (context) => {
			recorded.toolPhase.push(labelOf(context.context));
			return undefined;
		},
		finishTurn: (turn) => {
			recorded.sealPhase.push(labelOf(turn.context));
		},
		prepareNextTurnWithContext: async (turn) => {
			recorded.prepared++;
			// The genuine completed turn, not something rebuilt from the transcript.
			expect(turn.toolResults.length).toBe(1);
			expect(turn.message.content.some((block) => block.type === "toolCall")).toBe(true);
			if (options.prepareOnce && recorded.prepared > 1) return undefined;
			return { context: { ...turn.context, label: "prepared" } as Labelled, model: PREPARED_MODEL as never };
		},
		streamFn: (model) => {
			calls++;
			recorded.prompts.push(promptOf(model));
			recorded.models.push(model as unknown as string);
			const stream = createAssistantMessageEventStream();
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

/** One model call, its tool calls, and the seal. */
async function driveStep(agent: Agent): Promise<void> {
	const model = await agent.modelCall();
	const results = [];
	for (const call of model.toolCalls) {
		const result = await agent.runToolCall(call.id);
		if (result) results.push(result);
	}
	await agent.sealStep(results);
}

describe("the step cursor", () => {
	it("carries what preparation returned into the calls and the seal of that step", async () => {
		const { agent, recorded } = agentUnderTest();
		// Step one prepares nothing and its tool runs under the original prompt.
		await driveStep(agent);
		expect(recorded.toolPhase).toEqual(["original"]);
		expect(recorded.sealPhase).toEqual(["original"]);

		// Step two prepares. Everything that step does afterwards belongs to what it prepared, not
		// to the agent's state, which still says `original`.
		const model = await agent.modelCall();
		expect(recorded.prepared).toBe(1);
		expect(recorded.prompts).toEqual(["original", "prepared"]);
		expect(recorded.models[1]).toBe(PREPARED_MODEL);
		expect(agent.state.model).not.toBe(PREPARED_MODEL);

		const result = await agent.runToolCall(model.toolCalls[0].id);
		await agent.sealStep(result ? [result] : []);
		expect(recorded.toolPhase).toEqual(["original", "prepared"]);
		expect(recorded.sealPhase).toEqual(["original", "prepared"]);
	});

	it("does not prepare or call the provider again for a replayed response", async () => {
		const { agent, recorded } = agentUnderTest();
		const first = await agent.modelCall();
		const replay = await agent.modelCall();

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
				context: { messages: [], tools: [] },
				newMessages: [],
			} as unknown as PrepareNextTurnContext,
		};
		const context: AgentContext = {
			messages: [{ role: "user", content: "run", timestamp: Date.now() }],
			tools: [],
		};
		const config = {
			convertToLlm: (messages: AgentMessage[]) => messages.filter((m) => LLM_ROLES.includes(m.role)),
			prepareNextTurn: async () => {
				prepared++;
				return { context: { ...context }, model: PREPARED_MODEL };
			},
		} as unknown as AgentLoopConfig;
		const streamFn = ((model: unknown) => {
			prompts.push(promptOf(model));
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
		await driveStep(agent);
		expect(recorded.prepared).toBe(0);

		// A new run is not a continuation of the step before it: the loop begins with no completed
		// turn, so preparation must not fire on the run's own first model call.
		agent.state.messages = [{ role: "user", content: "again", timestamp: Date.now() }];
		await agent.prompt("again");

		expect(recorded.prompts[1]).toBe("original");
	});
	it("keeps the model preparation chose when a later preparation returns nothing", async () => {
		const { agent, recorded } = agentUnderTest({ prepareOnce: true });
		await driveStep(agent);
		await driveStep(agent);
		await driveStep(agent);

		// The loop keeps a replacement for the rest of the run, and so does a stepped turn.
		expect(recorded.prepared).toBe(2);
		expect(recorded.models.slice(1)).toEqual([PREPARED_MODEL, PREPARED_MODEL]);
	});

	// The loop carries what `prepareRequest` returned into every later request of the run. Stepped
	// from outside, the turn has to do the same, for the model and thinking level and for the
	// replaced context the tools run against.
	it("keeps what prepareRequest returned once for the rest of the turn, as the loop does", async () => {
		const run = async (stepped: boolean) => {
			const requests: string[] = [];
			const ran: string[] = [];
			let responses = 0;
			const tool = (impl: string) =>
				({
					...echoTool,
					execute: async () => {
						ran.push(impl);
						return { content: [{ type: "text", text: impl }], details: {} };
					},
				}) as AgentTool;
			const agent = new Agent({
				initialState: { tools: [tool("original")] },
				prepareRequest: ({ context, model }) => {
					if (requests.length > 0) return undefined;
					return {
						model: { ...model, id: "routed" },
						thinkingLevel: "high",
						context: { ...context, tools: [tool("replaced")] },
					};
				},
				streamFn: (model, _context, options) => {
					requests.push(`${(model as { id?: string }).id ?? "original"}/${options?.reasoning ?? "off"}`);
					responses++;
					const more = responses <= 2;
					const stream = createAssistantMessageEventStream();
					const message: AssistantMessage = {
						role: "assistant",
						content: more
							? [{ type: "toolCall", id: `call-${responses}`, name: "echo", arguments: { value: "x" } }]
							: [{ type: "text", text: "done" }],
						api: "openai-responses",
						provider: "openai",
						model: "mock",
						timestamp: Date.now(),
						stopReason: more ? "toolUse" : "stop",
						usage: {
							input: 0,
							output: 0,
							cacheRead: 0,
							cacheWrite: 0,
							totalTokens: 0,
							cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
						},
					};
					queueMicrotask(() => stream.push({ type: "done", reason: more ? "toolUse" : "stop", message }));
					return stream;
				},
			});
			if (stepped) {
				agent.state.messages = [{ role: "user", content: "run", timestamp: Date.now() }];
				for (let step = 0; step < 3; step++) await driveStep(agent);
			} else {
				await agent.prompt("run");
			}
			return { requests, ran };
		};

		const ordinary = await run(false);
		expect(ordinary.requests).toEqual(["routed/high", "routed/high", "routed/high"]);
		expect(ordinary.ran).toEqual(["replaced", "replaced"]);
		expect(await run(true)).toEqual(ordinary);
	});

	it("forgets what the last run's steps left behind on reset() and startTurn()", async () => {
		for (const fresh of [(agent: Agent) => agent.reset(), (agent: Agent) => agent.startTurn()]) {
			const { agent, recorded } = agentUnderTest();
			await driveStep(agent);
			fresh(agent);
			agent.state.messages = [{ role: "user", content: "new task", timestamp: Date.now() }];

			// A new run starts from no completed turn, so it does not prepare from the old task.
			await agent.modelCall();
			expect(recorded.prepared).toBe(0);
			expect(recorded.prompts).toEqual(["original", "original"]);
		}
	});

	it("reports a response that ended the run again, and refuses an empty transcript", async () => {
		const { agent, recorded } = agentUnderTest();
		await driveStep(agent);
		const isAssistant = (m: AgentMessage): m is AssistantMessage => m.role === "assistant";
		const asked = agent.state.messages.findLast(isAssistant)!;
		const failed: AssistantMessage = { ...asked, content: [], stopReason: "error" };
		agent.state.messages = [...agent.state.messages, failed];
		const before = agent.state.messages.length;

		// Reported again, not refused, so a caller that lost the first answer still seals it. Nothing
		// is asked or recorded, since a new failed message would read as one more failed attempt.
		expect(await agent.modelCall()).toEqual({
			toolCalls: [],
			sequential: false,
			ended: true,
			stepId: expect.any(String),
		});
		expect(agent.state.messages).toHaveLength(before);
		expect(agent.state.isStreaming).toBe(false);

		agent.state.messages = [];
		await expect(agent.modelCall()).rejects.toThrow("No messages to step from");
		expect(agent.state.messages).toEqual([]);
		expect(recorded.prompts).toEqual(["original"]);
	});
});
