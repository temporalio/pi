import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, ToolResultMessage, UserMessage } from "@earendil-works/pi-ai/compat";
import { describe, expect, it } from "vitest";
import { findDanglingToolCalls } from "../../src/core/session-manager.ts";

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

function toolResult(toolCallId: string): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId,
		toolName: "do",
		content: [{ type: "text", text: "done" }],
		details: {},
		isError: false,
		timestamp: Date.now(),
	};
}

function user(text: string): UserMessage {
	return { role: "user", content: [{ type: "text", text }], timestamp: Date.now() };
}

const call = (id: string) => ({ type: "toolCall" as const, id, name: "do", arguments: {} });

describe("findDanglingToolCalls", () => {
	it("returns nothing for an empty transcript", () => {
		expect(findDanglingToolCalls([])).toEqual([]);
	});

	it("returns nothing when every tool call has a result", () => {
		const messages: AgentMessage[] = [
			user("go"),
			assistant([call("t1")], "toolUse"),
			toolResult("t1"),
			assistant([{ type: "text", text: "all set" }]),
		];
		expect(findDanglingToolCalls(messages)).toEqual([]);
	});

	it("finds every call of the trailing assistant message", () => {
		const messages: AgentMessage[] = [user("go"), assistant([call("t1"), call("t2")], "toolUse")];
		expect(findDanglingToolCalls(messages).map((c) => c.id)).toEqual(["t1", "t2"]);
	});

	it("leaves an unresolved call from earlier history alone", () => {
		// An aborted tool batch leaves calls open on purpose. A result appended at the
		// tail would attach to the wrong call.
		const messages: AgentMessage[] = [
			user("go"),
			assistant([call("old")], "toolUse"),
			user("never mind, do this instead"),
			assistant([{ type: "text", text: "sure" }]),
		];
		expect(findDanglingToolCalls(messages)).toEqual([]);
	});

	// The provider never sees an errored or aborted message, so a result for its calls would have
	// nothing to pair with.
	it.each(["aborted", "error"] as const)("skips an assistant message that stopped on %s", (stopReason) => {
		const messages: AgentMessage[] = [user("go"), assistant([call("t1")], stopReason)];
		expect(findDanglingToolCalls(messages)).toEqual([]);
	});
});
