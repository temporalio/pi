/**
 * Agent loop that works with AgentMessage throughout.
 * Transforms to Message[] only at the LLM call boundary.
 */

import {
	type AssistantMessage,
	type Context,
	EventStream,
	type ToolResultMessage,
	validateToolArguments,
} from "@earendil-works/pi-ai";
import { getDefaultStreamFn } from "./stream-fn.ts";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentTool,
	AgentToolCall,
	AgentToolResult,
	StreamFn,
} from "./types.ts";

export type AgentEventSink = (event: AgentEvent) => Promise<void> | void;

/**
 * Start an agent loop with a new prompt message.
 * The prompt is added to the context and events are emitted for it.
 */
export function agentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	streamFn: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	const stream = createAgentStream();

	void runAgentLoop(
		prompts,
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

/**
 * Continue an agent loop from the current context without adding a new message.
 * Used for retries - context already has user message or tool results.
 *
 * **Important:** The last message in context must convert to a `user` or `toolResult` message
 * via `convertToLlm`. If it doesn't, the LLM provider will reject the request.
 * This cannot be validated here since `convertToLlm` is only called once per turn.
 */
export function agentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	streamFn: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const stream = createAgentStream();

	void runAgentLoopContinue(
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

export async function runAgentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal: AbortSignal | undefined,
	streamFn: StreamFn,
): Promise<AgentMessage[]> {
	const newMessages: AgentMessage[] = [...prompts];
	const currentContext: AgentContext = {
		...context,
		messages: [...context.messages, ...prompts],
	};

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });
	for (const prompt of prompts) {
		await emit({ type: "message_start", message: prompt });
		await emit({ type: "message_end", message: prompt });
	}

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn ?? getDefaultStreamFn());
	return newMessages;
}

export async function runAgentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal: AbortSignal | undefined,
	streamFn: StreamFn,
): Promise<AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const newMessages: AgentMessage[] = [];
	const currentContext: AgentContext = { ...context };

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn ?? getDefaultStreamFn());
	return newMessages;
}

export interface AgentStepOutcome {
	messages: AgentMessage[];
	/** True when the turn ran tools whose results still need another step. */
	hasMoreToolCalls: boolean;
}

/**
 * Run exactly one iteration of the loop against the current context, adding no
 * new message. Like agentLoopContinue, the last message must convert to a `user`
 * or `toolResult` message. Used to drive a turn one step at a time from outside.
 *
 * A step is one turn, not a whole run, so it emits no `agent_start`. The run ends
 * only when the turn itself ends it, on an error, an abort, or a stop decision.
 */
export function agentStep(
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	streamFn: StreamFn,
): EventStream<AgentEvent, AgentStepOutcome> {
	if (context.messages.length === 0) {
		throw new Error("Cannot step: no messages in context");
	}
	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot step from message role: assistant");
	}

	// A step has no single terminating event, so the stream closes when the step
	// resolves. That also carries the outcome, which no event holds.
	const stream = new EventStream<AgentEvent, AgentStepOutcome>(
		() => false,
		() => ({ messages: [], hasMoreToolCalls: false }),
	);

	void runAgentStep(
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((outcome) => {
		stream.end(outcome);
	});

	return stream;
}

export async function runAgentStep(
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal: AbortSignal | undefined,
	streamFn: StreamFn,
): Promise<AgentStepOutcome> {
	if (context.messages.length === 0) {
		throw new Error("Cannot step: no messages in context");
	}
	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot step from message role: assistant");
	}

	const newMessages: AgentMessage[] = [];
	const currentContext: AgentContext = { ...context };

	const outcome = await runSingleTurn({
		context: currentContext,
		config,
		newMessages,
		pendingMessages: [],
		emitTurnStart: true,
		fetchNextPending: false,
		signal,
		emit,
		streamFunction: streamFn ?? getDefaultStreamFn(),
	});

	return { messages: newMessages, hasMoreToolCalls: !outcome.done && outcome.hasMoreToolCalls };
}

export interface AgentModelCallOutcome {
	/** The calls the model asked for, in the order it asked for them. */
	toolCalls: AgentToolCall[];
	/** Whether the calls have to run one at a time. */
	sequential: boolean;
	/**
	 * The response ended the run on its own (an error or an abort). There is nothing to
	 * dispatch and nothing to seal.
	 */
	ended: boolean;
	/**
	 * The transcript already held the response, so no model call was made. A driver whose
	 * record of the call was lost gets the same answer back instead of paying for it twice,
	 * and the calls it reports are ones nothing has run yet.
	 */
	replayed: boolean;
}

/**
 * The model call of one step, on its own. The calls it reports are recorded but not run, so a
 * caller can put each of them somewhere the loop cannot see: its own unit of work, its own
 * retry policy, its own approval.
 */
export async function runAgentModelCall(
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal: AbortSignal | undefined,
	streamFn: StreamFn,
): Promise<AgentModelCallOutcome> {
	if (context.messages.length === 0) {
		throw new Error("Cannot step: no messages in context");
	}

	const last = context.messages[context.messages.length - 1];
	if (last.role === "assistant") {
		// A response with nothing to act on is not one a step can be resumed from; the caller
		// settles it (prepareStep does) before asking for another.
		if (last.stopReason === "error" || last.stopReason === "aborted") {
			throw new Error("Cannot step from message role: assistant");
		}
		// The seal that follows emits turn_end either way, so a replayed step has to open the turn
		// too. An extension pairing the two would see the boundaries drift apart otherwise.
		await emit({ type: "turn_start" });
		const toolCalls = last.content.filter((c) => c.type === "toolCall");
		return {
			toolCalls,
			sequential: mustRunToolCallsInOrder(context, config, last, toolCalls),
			ended: false,
			replayed: true,
		};
	}

	const outcome = await runTurnModelCall({
		context: { ...context },
		config,
		newMessages: [],
		pendingMessages: [],
		emitTurnStart: true,
		signal,
		emit,
		streamFunction: streamFn ?? getDefaultStreamFn(),
	});

	return {
		toolCalls: outcome.toolCalls,
		sequential: mustRunToolCallsInOrder(context, config, outcome.message, outcome.toolCalls),
		ended: outcome.ended,
		replayed: false,
	};
}

/**
 * Run one recorded call of the current step. Returns undefined when the transcript already
 * holds a result for it, which is the at-least-once case: nothing runs a second time.
 */
export async function runAgentToolCall(
	context: AgentContext,
	config: AgentLoopConfig,
	toolCallId: string,
	emit: AgentEventSink,
	signal: AbortSignal | undefined,
): Promise<TurnToolCallOutcome | undefined> {
	if (context.messages.some((m) => m.role === "toolResult" && m.toolCallId === toolCallId)) {
		return undefined;
	}

	const assistantMessage = lastAssistantMessage(context.messages);
	const toolCall = assistantMessage?.content.find(
		(c): c is AgentToolCall => c.type === "toolCall" && c.id === toolCallId,
	);
	if (!assistantMessage || !toolCall) {
		throw new Error(`No recorded tool call ${toolCallId} to run`);
	}

	return runTurnToolCall({ context: { ...context }, assistantMessage, toolCall, config, signal, emit });
}

/**
 * Close the current step with the results of its calls, in the order the model asked for them.
 * The step's message is the last assistant message, so a seal that runs twice finds the same
 * one and records only what is missing.
 */
export async function runAgentSeal(
	context: AgentContext,
	config: AgentLoopConfig,
	toolCalls: ReadonlyArray<TurnToolCallOutcome>,
	emit: AgentEventSink,
): Promise<AgentStepOutcome> {
	const message = lastAssistantMessage(context.messages);
	if (!message) {
		throw new Error("No assistant message to seal");
	}

	// A response that ended the run closed the turn as it went, so there is nothing left to
	// record and nothing to decide. The caller still seals, because what happens after a
	// failed model call (a retry, a compaction) is above the loop.
	if (message.stopReason === "error" || message.stopReason === "aborted") {
		return { messages: [], hasMoreToolCalls: false };
	}

	const newMessages: AgentMessage[] = [];
	const outcome = await sealTurnStep({
		context: { ...context },
		config,
		newMessages,
		message,
		toolCalls,
		fetchNextPending: false,
		emit,
	});

	return { messages: newMessages, hasMoreToolCalls: !outcome.done && outcome.hasMoreToolCalls };
}

/**
 * What the transcript says about a call when nothing can say whether it ran. The tool can have
 * had its effect before the run stopped, so calling it a failure would invite a second run of
 * something that already happened.
 */
export const UNKNOWN_TOOL_CALL_OUTCOME =
	"The outcome of this tool call is unknown. The session stopped after the call started " +
	"but before the result was recorded. It can have taken effect. Check the current state " +
	"before you try again.";

/** Settle a call nothing can answer for, so the step it belongs to still closes. */
export function unknownToolCallOutcome(toolCall: { id: string; name: string }): TurnToolCallOutcome {
	return {
		message: {
			role: "toolResult",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			content: [{ type: "text", text: UNKNOWN_TOOL_CALL_OUTCOME }],
			details: {},
			isError: true,
			timestamp: Date.now(),
		},
		terminate: false,
	};
}

function lastAssistantMessage(messages: ReadonlyArray<AgentMessage>): AssistantMessage | undefined {
	for (let index = messages.length - 1; index >= 0; index--) {
		const message = messages[index];
		if (message.role === "assistant") {
			return message;
		}
	}
	return undefined;
}

function createAgentStream(): EventStream<AgentEvent, AgentMessage[]> {
	return new EventStream<AgentEvent, AgentMessage[]>(
		(event: AgentEvent) => event.type === "agent_end",
		(event: AgentEvent) => (event.type === "agent_end" ? event.messages : []),
	);
}

interface SingleTurnParams {
	context: AgentContext;
	config: AgentLoopConfig;
	newMessages: AgentMessage[];
	pendingMessages: AgentMessage[];
	// The first iteration of a run does not emit turn_start; the run entry point
	// already emitted it.
	emitTurnStart: boolean;
	// Whether to drain the steering queue after the turn. A one-shot step leaves
	// the queues to the caller instead of consuming a message it will not run.
	fetchNextPending: boolean;
	signal: AbortSignal | undefined;
	emit: AgentEventSink;
	streamFunction: StreamFn;
}

interface SingleTurnOutcome {
	// True once agent_end has been emitted (an error/abort or a stop decision); the
	// caller must return without emitting agent_end again.
	done: boolean;
	hasMoreToolCalls: boolean;
	context: AgentContext;
	config: AgentLoopConfig;
	pendingMessages: AgentMessage[];
}

export interface TurnModelCallParams {
	context: AgentContext;
	config: AgentLoopConfig;
	newMessages: AgentMessage[];
	pendingMessages: AgentMessage[];
	emitTurnStart: boolean;
	signal: AbortSignal | undefined;
	emit: AgentEventSink;
	streamFunction: StreamFn;
}

export interface TurnModelCallOutcome {
	message: AssistantMessage;
	/** The calls the model asked for, in the order it asked for them. */
	toolCalls: AgentToolCall[];
	/**
	 * The response ended the run on its own (an error or an abort), and agent_end has
	 * been emitted. Nothing may be dispatched and nothing may be sealed.
	 */
	ended: boolean;
	context: AgentContext;
}

export interface TurnToolCallParams {
	context: AgentContext;
	/** The message that asked for this call. Its stop reason decides whether the call may run. */
	assistantMessage: AssistantMessage;
	toolCall: AgentToolCall;
	config: AgentLoopConfig;
	signal: AbortSignal | undefined;
	emit: AgentEventSink;
}

export interface TurnToolCallOutcome {
	message: ToolResultMessage;
	/** The tool asked for the run to stop. A batch ends the turn only when every call does. */
	terminate: boolean;
}

export interface SealTurnStepParams {
	context: AgentContext;
	config: AgentLoopConfig;
	newMessages: AgentMessage[];
	/** The message this step opened. */
	message: AssistantMessage;
	/** The step's settled calls, in the order the model asked for them. */
	toolCalls: ReadonlyArray<TurnToolCallOutcome>;
	fetchNextPending: boolean;
	emit: AgentEventSink;
}

/**
 * The model call of one step: inject any pending messages and stream a single assistant
 * response, stopping before the tools it asks for. The caller runs them.
 */
export async function runTurnModelCall(params: TurnModelCallParams): Promise<TurnModelCallOutcome> {
	const { newMessages, signal, emit, streamFunction: streamFn } = params;
	const context = params.context;

	if (params.emitTurnStart) {
		await emit({ type: "turn_start" });
	}

	for (const message of params.pendingMessages) {
		await emit({ type: "message_start", message });
		await emit({ type: "message_end", message });
		context.messages.push(message);
		newMessages.push(message);
	}

	const message = await streamAssistantResponse(context, params.config, signal, emit, streamFn);
	newMessages.push(message);

	if (message.stopReason === "error" || message.stopReason === "aborted") {
		await emit({ type: "turn_end", message, toolResults: [] });
		await emit({ type: "agent_end", messages: newMessages });
		return { message, toolCalls: [], ended: true, context };
	}

	return {
		message,
		toolCalls: message.content.filter((c) => c.type === "toolCall"),
		ended: false,
		context,
	};
}

/**
 * One recorded tool call, start to finish. It reports its result rather than entering it in
 * the transcript, because the results of a step go in together, in the order the model asked
 * for them, and a caller running calls concurrently settles them out of order.
 */
export async function runTurnToolCall(params: TurnToolCallParams): Promise<TurnToolCallOutcome> {
	const { context, assistantMessage, toolCall, config, signal, emit } = params;

	await emit({
		type: "tool_execution_start",
		toolCallId: toolCall.id,
		toolName: toolCall.name,
		args: toolCall.arguments,
	});

	// A "length" stop means the output was cut off by the token limit, so every tool call in
	// the message may carry truncated arguments. Fail it instead of executing a borked call.
	const finalized =
		assistantMessage.stopReason === "length"
			? truncatedToolCallOutcome(toolCall)
			: await settleToolCall(context, assistantMessage, toolCall, config, signal, emit);

	await emitToolExecutionEnd(finalized, emit);
	return toTurnToolCallOutcome(finalized);
}

/**
 * Close a step whose calls have settled: record their results, then decide whether the turn
 * keeps going. A step that ran no tools seals the same way.
 */
export async function sealTurnStep(params: SealTurnStepParams): Promise<SingleTurnOutcome> {
	const { newMessages, emit, message } = params;
	let currentContext = params.context;
	let config = params.config;

	// A seal cut short can have recorded some of the results already, so each is checked on its
	// own. The batch decides the turn either way: dropping a recorded result from the count
	// would end a turn that has more to do.
	const recorded = new Set(currentContext.messages.filter((m) => m.role === "toolResult").map((m) => m.toolCallId));
	const toolResults: ToolResultMessage[] = [];
	for (const call of params.toolCalls) {
		toolResults.push(call.message);
		if (recorded.has(call.message.toolCallId)) {
			continue;
		}
		await emitToolResultMessage(call.message, emit);
		currentContext.messages.push(call.message);
		newMessages.push(call.message);
	}
	const hasMoreToolCalls = params.toolCalls.length > 0 && !shouldTerminateToolBatch(params.toolCalls);

	await emit({ type: "turn_end", message, toolResults });

	const nextTurnContext = {
		message,
		toolResults,
		context: currentContext,
		newMessages,
	};
	const nextTurnSnapshot = await config.prepareNextTurn?.(nextTurnContext);
	if (nextTurnSnapshot) {
		currentContext = nextTurnSnapshot.context ?? currentContext;
		config = {
			...config,
			model: nextTurnSnapshot.model ?? config.model,
			reasoning:
				nextTurnSnapshot.thinkingLevel === undefined
					? config.reasoning
					: nextTurnSnapshot.thinkingLevel === "off"
						? undefined
						: nextTurnSnapshot.thinkingLevel,
		};
	}

	if (
		await config.shouldStopAfterTurn?.({
			message,
			toolResults,
			context: currentContext,
			newMessages,
		})
	) {
		await emit({ type: "agent_end", messages: newMessages });
		return { done: true, hasMoreToolCalls, context: currentContext, config, pendingMessages: [] };
	}

	const steering = params.fetchNextPending ? await config.getSteeringMessages?.() : undefined;
	const pendingMessages = steering || [];
	return { done: false, hasMoreToolCalls, context: currentContext, config, pendingMessages };
}

/**
 * One iteration of the agent loop: inject any pending messages, stream a single
 * assistant response, run the tools it requests, and report whether the loop
 * should keep going.
 *
 * The three pieces are the same ones a caller stepping from outside drives, so a turn under
 * an executor and a turn pi runs itself are one implementation.
 */
async function runSingleTurn(params: SingleTurnParams): Promise<SingleTurnOutcome> {
	const modelCall = await runTurnModelCall(params);
	if (modelCall.ended) {
		return {
			done: true,
			hasMoreToolCalls: false,
			context: modelCall.context,
			config: params.config,
			pendingMessages: [],
		};
	}

	const toolCalls = await dispatchToolCalls(
		modelCall.context,
		modelCall.message,
		modelCall.toolCalls,
		params.config,
		params.signal,
		params.emit,
	);

	return sealTurnStep({
		context: modelCall.context,
		config: params.config,
		newMessages: params.newMessages,
		message: modelCall.message,
		toolCalls,
		fetchNextPending: params.fetchNextPending,
		emit: params.emit,
	});
}

/**
 * Main loop logic shared by agentLoop and agentLoopContinue.
 */
async function runLoop(
	initialContext: AgentContext,
	newMessages: AgentMessage[],
	initialConfig: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFunction: StreamFn,
): Promise<void> {
	let currentContext = initialContext;
	let config = initialConfig;
	let firstTurn = true;
	// Check for steering messages at start (user may have typed while waiting)
	let pendingMessages: AgentMessage[] = (await config.getSteeringMessages?.()) || [];

	// Outer loop: continues when queued follow-up messages arrive after agent would stop
	while (true) {
		let hasMoreToolCalls = true;

		// Inner loop: process tool calls and steering messages
		while (hasMoreToolCalls || pendingMessages.length > 0) {
			const outcome = await runSingleTurn({
				context: currentContext,
				config,
				newMessages,
				pendingMessages,
				emitTurnStart: !firstTurn,
				fetchNextPending: true,
				signal,
				emit,
				streamFunction,
			});
			firstTurn = false;
			if (outcome.done) {
				return;
			}
			hasMoreToolCalls = outcome.hasMoreToolCalls;
			currentContext = outcome.context;
			config = outcome.config;
			pendingMessages = outcome.pendingMessages;
		}

		// Agent would stop here. Check for follow-up messages.
		const followUpMessages = (await config.getFollowUpMessages?.()) || [];
		if (followUpMessages.length > 0) {
			// Set as pending so inner loop processes them
			pendingMessages = followUpMessages;
			continue;
		}

		// No more messages, exit
		break;
	}

	await emit({ type: "agent_end", messages: newMessages });
}

/**
 * Stream an assistant response from the LLM.
 * This is where AgentMessage[] gets transformed to Message[] for the LLM.
 */
async function streamAssistantResponse(
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFunction: StreamFn,
): Promise<AssistantMessage> {
	// Apply context transform if configured (AgentMessage[] → AgentMessage[])
	let messages = context.messages;
	if (config.transformContext) {
		messages = await config.transformContext(messages, signal);
	}

	// Convert to LLM-compatible messages (AgentMessage[] → Message[])
	const llmMessages = await config.convertToLlm(messages);

	// Build LLM context
	const llmContext: Context = {
		systemPrompt: context.systemPrompt,
		messages: llmMessages,
		tools: context.tools,
	};

	// Resolve API key (important for expiring tokens)
	const resolvedApiKey =
		(config.getApiKey ? await config.getApiKey(config.model.provider) : undefined) || config.apiKey;

	const response = await streamFunction(config.model, llmContext, {
		...config,
		apiKey: resolvedApiKey,
		signal,
	});

	let partialMessage: AssistantMessage | null = null;
	let addedPartial = false;

	for await (const event of response) {
		switch (event.type) {
			case "start":
				partialMessage = event.partial;
				context.messages.push(partialMessage);
				addedPartial = true;
				await emit({ type: "message_start", message: { ...partialMessage } });
				break;

			case "text_start":
			case "text_delta":
			case "text_end":
			case "thinking_start":
			case "thinking_delta":
			case "thinking_end":
			case "toolcall_start":
			case "toolcall_delta":
			case "toolcall_end":
				if (partialMessage) {
					partialMessage = event.partial;
					context.messages[context.messages.length - 1] = partialMessage;
					await emit({
						type: "message_update",
						assistantMessageEvent: event,
						message: { ...partialMessage },
					});
				}
				break;

			case "done":
			case "error": {
				const finalMessage = await response.result();
				if (addedPartial) {
					context.messages[context.messages.length - 1] = finalMessage;
				} else {
					context.messages.push(finalMessage);
				}
				if (!addedPartial) {
					await emit({ type: "message_start", message: { ...finalMessage } });
				}
				await emit({ type: "message_end", message: finalMessage });
				return finalMessage;
			}
		}
	}

	const finalMessage = await response.result();
	if (addedPartial) {
		context.messages[context.messages.length - 1] = finalMessage;
	} else {
		context.messages.push(finalMessage);
		await emit({ type: "message_start", message: { ...finalMessage } });
	}
	await emit({ type: "message_end", message: finalMessage });
	return finalMessage;
}

/**
 * The result a call gets when the response that asked for it was cut off by the output
 * token limit. Streamed tool-call arguments are finalized with a best-effort JSON salvage
 * parser, so a truncated message can yield tool calls whose arguments parse and validate but
 * are silently incomplete. None of them are safe to execute; report each as an error so the
 * model can re-issue it.
 */
function truncatedToolCallOutcome(toolCall: AgentToolCall): FinalizedToolCallOutcome {
	return {
		toolCall,
		result: createErrorToolResult(
			`Tool call "${toolCall.name}" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.`,
		),
		isError: true,
	};
}

/** Prepare, run and finalize one call, without deciding anything about the batch it is in. */
async function settleToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCall: AgentToolCall,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<FinalizedToolCallOutcome> {
	const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
	if (preparation.kind === "immediate") {
		return { toolCall, result: preparation.result, isError: preparation.isError };
	}
	const executed = await executePreparedToolCall(preparation, signal, emit);
	return finalizeExecutedToolCall(currentContext, assistantMessage, preparation, executed, config, signal);
}

function toTurnToolCallOutcome(finalized: FinalizedToolCallOutcome): TurnToolCallOutcome {
	return {
		message: createToolResultMessage(finalized),
		terminate: finalized.result.terminate === true,
	};
}

/**
 * Run the calls of one step and report them in the order the model asked for them.
 */
async function dispatchToolCalls(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<TurnToolCallOutcome[]> {
	if (toolCalls.length === 0) {
		return [];
	}
	if (mustRunToolCallsInOrder(currentContext, config, assistantMessage, toolCalls)) {
		return dispatchToolCallsSequential(currentContext, assistantMessage, toolCalls, config, signal, emit);
	}
	return dispatchToolCallsParallel(currentContext, assistantMessage, toolCalls, config, signal, emit);
}

/**
 * Whether the calls of one step have to run one at a time. A caller that runs them elsewhere
 * has to ask before it fans them out, so the answer belongs to the loop rather than to it.
 */
export function mustRunToolCallsInOrder(
	context: AgentContext,
	config: AgentLoopConfig,
	assistantMessage: AssistantMessage,
	toolCalls: ReadonlyArray<AgentToolCall>,
): boolean {
	// A truncated response runs nothing, so there is no execution to overlap.
	if (assistantMessage.stopReason === "length" || config.toolExecution === "sequential") {
		return true;
	}
	return toolCalls.some((tc) => context.tools?.find((t) => t.name === tc.name)?.executionMode === "sequential");
}

async function dispatchToolCallsSequential(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<TurnToolCallOutcome[]> {
	const outcomes: TurnToolCallOutcome[] = [];

	for (const toolCall of toolCalls) {
		outcomes.push(
			await runTurnToolCall({ context: currentContext, assistantMessage, toolCall, config, signal, emit }),
		);
		// An abort leaves the rest of the batch unsettled on purpose: the calls are still in the
		// transcript, and settling them here would answer for tools that never ran.
		if (signal?.aborted) {
			break;
		}
	}

	return outcomes;
}

async function dispatchToolCallsParallel(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<TurnToolCallOutcome[]> {
	const finalizedCalls: FinalizedToolCallEntry[] = [];

	// Preparation stays in the model's order, because a permission ask is a preparation and
	// asking about four tools at once is not a question anyone can answer. Only execution overlaps.
	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		if (preparation.kind === "immediate") {
			const finalized = {
				toolCall,
				result: preparation.result,
				isError: preparation.isError,
			} satisfies FinalizedToolCallOutcome;
			await emitToolExecutionEnd(finalized, emit);
			finalizedCalls.push(finalized);
			if (signal?.aborted) {
				break;
			}
			continue;
		}

		finalizedCalls.push(async () => {
			const executed = await executePreparedToolCall(preparation, signal, emit);
			const finalized = await finalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				preparation,
				executed,
				config,
				signal,
			);
			await emitToolExecutionEnd(finalized, emit);
			return finalized;
		});
		if (signal?.aborted) {
			break;
		}
	}

	const orderedFinalizedCalls = await Promise.all(
		finalizedCalls.map((entry) => (typeof entry === "function" ? entry() : Promise.resolve(entry))),
	);
	return orderedFinalizedCalls.map(toTurnToolCallOutcome);
}

type PreparedToolCall = {
	kind: "prepared";
	toolCall: AgentToolCall;
	tool: AgentTool<any>;
	args: unknown;
};

type ImmediateToolCallOutcome = {
	kind: "immediate";
	result: AgentToolResult<any>;
	isError: boolean;
};

type ExecutedToolCallOutcome = {
	result: AgentToolResult<any>;
	isError: boolean;
};

type FinalizedToolCallOutcome = {
	toolCall: AgentToolCall;
	result: AgentToolResult<any>;
	isError: boolean;
};

type FinalizedToolCallEntry = FinalizedToolCallOutcome | (() => Promise<FinalizedToolCallOutcome>);

function shouldTerminateToolBatch(calls: ReadonlyArray<{ terminate: boolean }>): boolean {
	return calls.length > 0 && calls.every((call) => call.terminate);
}

function prepareToolCallArguments(tool: AgentTool<any>, toolCall: AgentToolCall): AgentToolCall {
	if (!tool.prepareArguments) {
		return toolCall;
	}
	const preparedArguments = tool.prepareArguments(toolCall.arguments);
	if (preparedArguments === toolCall.arguments) {
		return toolCall;
	}
	return {
		...toolCall,
		arguments: preparedArguments as Record<string, any>,
	};
}

async function prepareToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCall: AgentToolCall,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<PreparedToolCall | ImmediateToolCallOutcome> {
	const tool = currentContext.tools?.find((t) => t.name === toolCall.name);
	if (!tool) {
		return {
			kind: "immediate",
			result: createErrorToolResult(`Tool ${toolCall.name} not found`),
			isError: true,
		};
	}

	try {
		const preparedToolCall = prepareToolCallArguments(tool, toolCall);
		const validatedArgs = validateToolArguments(tool, preparedToolCall);
		if (config.beforeToolCall) {
			const beforeResult = await config.beforeToolCall(
				{
					assistantMessage,
					toolCall,
					args: validatedArgs,
					context: currentContext,
				},
				signal,
			);
			if (signal?.aborted) {
				return {
					kind: "immediate",
					result: createErrorToolResult("Operation aborted"),
					isError: true,
				};
			}
			if (beforeResult?.block) {
				const result = createErrorToolResult(beforeResult.reason || "Tool execution was blocked");
				if (beforeResult.terminate === true) {
					result.terminate = true;
				}
				return {
					kind: "immediate",
					result,
					isError: true,
				};
			}
		}
		if (signal?.aborted) {
			return {
				kind: "immediate",
				result: createErrorToolResult("Operation aborted"),
				isError: true,
			};
		}
		return {
			kind: "prepared",
			toolCall,
			tool,
			args: validatedArgs,
		};
	} catch (error) {
		return {
			kind: "immediate",
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

async function executePreparedToolCall(
	prepared: PreparedToolCall,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallOutcome> {
	const updateEvents: Promise<void>[] = [];
	let acceptingUpdates = true;

	try {
		const result = await prepared.tool.execute(
			prepared.toolCall.id,
			prepared.args as never,
			signal,
			(partialResult) => {
				if (!acceptingUpdates) return;
				updateEvents.push(
					Promise.resolve(
						emit({
							type: "tool_execution_update",
							toolCallId: prepared.toolCall.id,
							toolName: prepared.toolCall.name,
							args: prepared.toolCall.arguments,
							partialResult,
						}),
					),
				);
			},
		);
		acceptingUpdates = false;
		await Promise.all(updateEvents);
		return { result, isError: false };
	} catch (error) {
		acceptingUpdates = false;
		await Promise.all(updateEvents);
		return {
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	} finally {
		acceptingUpdates = false;
	}
}

async function finalizeExecutedToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	prepared: PreparedToolCall,
	executed: ExecutedToolCallOutcome,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<FinalizedToolCallOutcome> {
	let result = executed.result;
	let isError = executed.isError;

	if (config.afterToolCall) {
		try {
			const afterResult = await config.afterToolCall(
				{
					assistantMessage,
					toolCall: prepared.toolCall,
					args: prepared.args,
					result,
					isError,
					context: currentContext,
				},
				signal,
			);
			if (afterResult) {
				result = {
					...result,
					content: afterResult.content ?? result.content,
					details: afterResult.details ?? result.details,
					usage: afterResult.usage ?? result.usage,
					terminate: afterResult.terminate ?? result.terminate,
				};
				isError = afterResult.isError ?? isError;
			}
		} catch (error) {
			result = createErrorToolResult(error instanceof Error ? error.message : String(error));
			isError = true;
		}
	}

	return {
		toolCall: prepared.toolCall,
		result,
		isError,
	};
}

function createErrorToolResult(message: string): AgentToolResult<any> {
	return {
		content: [{ type: "text", text: message }],
		details: {},
	};
}

async function emitToolExecutionEnd(finalized: FinalizedToolCallOutcome, emit: AgentEventSink): Promise<void> {
	await emit({
		type: "tool_execution_end",
		toolCallId: finalized.toolCall.id,
		toolName: finalized.toolCall.name,
		result: finalized.result,
		isError: finalized.isError,
	});
}

function createToolResultMessage(finalized: FinalizedToolCallOutcome): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId: finalized.toolCall.id,
		toolName: finalized.toolCall.name,
		// Untyped tools (JS extensions) can return results without content; normalize
		// so the null never enters session history or provider payloads.
		content: finalized.result.content ?? [],
		details: finalized.result.details,
		usage: finalized.result.usage,
		...(finalized.result.addedToolNames?.length ? { addedToolNames: finalized.result.addedToolNames } : {}),
		isError: finalized.isError,
		timestamp: Date.now(),
	};
}

async function emitToolResultMessage(toolResultMessage: ToolResultMessage, emit: AgentEventSink): Promise<void> {
	await emit({ type: "message_start", message: toolResultMessage });
	await emit({ type: "message_end", message: toolResultMessage });
}
