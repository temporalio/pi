#!/usr/bin/env node

/**
 * Rewrites the agent core and coding agent packages in place so they publish as Temporal's
 * build under the `@temporalio` scope. Run it from the repo root after the build and before
 * `npm publish`.
 *
 * Usage: node scripts/temporal-publish.mjs <pi-temporal version> [--check]
 */

import { copyFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const REPOSITORY_URL = "git+https://github.com/temporalio/pi.git";
const README_MARKER = "<!-- temporal-build -->";
const README_NOTE = `${README_MARKER}
> This is Temporal's build of [earendil-works/pi](https://github.com/earendil-works/pi). It adds
> the step-level API that [pi-temporal](https://github.com/temporalio/pi-temporal) needs. The
> coding agent CLI is also called \`pi\`, so it conflicts with a global install of the upstream
> \`pi\`.

`;

const PACKAGES = [
	{ dir: "agent", upstreamName: "@earendil-works/pi-agent-core", name: "@temporalio/pi-agent-core" },
	{
		dir: "coding-agent",
		upstreamName: "@earendil-works/pi-coding-agent",
		name: "@temporalio/pi-coding-agent",
	},
];

const args = process.argv.slice(2);
const check = args.includes("--check");
const positional = args.filter((arg) => arg !== "--check");
if (positional.length !== 1 || args.some((arg) => arg.startsWith("--") && arg !== "--check")) {
	console.error("Usage: node scripts/temporal-publish.mjs <pi-temporal version> [--check]");
	process.exit(1);
}
const piTemporalVersion = positional[0];
// Tags carry a plain release version, and the build version must stay valid SemVer.
if (!/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(piTemporalVersion)) {
	console.error(`Expected a pi-temporal version such as 0.2.0, got "${piTemporalVersion}"`);
	process.exit(1);
}

const repoRoot = process.cwd();
const license = join(repoRoot, "LICENSE");
if (!existsSync(license)) throw new Error(`Run this from the repo root. ${license} is missing.`);

const packages = PACKAGES.map((spec) => {
	const directory = join(repoRoot, "packages", spec.dir);
	const manifestPath = join(directory, "package.json");
	const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
	// A second run sees the rewritten manifest, so accept both names and strip the old suffix.
	if (manifest.name !== spec.upstreamName && manifest.name !== spec.name) {
		throw new Error(`${manifestPath} is named ${manifest.name}, expected ${spec.upstreamName}`);
	}
	const upstreamVersion = manifest.version.split("-temporal.")[0];
	if (!/^\d+\.\d+\.\d+$/.test(upstreamVersion)) {
		throw new Error(`${manifestPath} has version ${manifest.version}, expected a release version`);
	}
	const version = `${upstreamVersion}-temporal.${piTemporalVersion}`;
	return { ...spec, directory, manifestPath, manifest, version };
});

const [agentCore, codingAgent] = packages;
const coreRange = codingAgent.manifest.dependencies?.[agentCore.upstreamName];
if (!coreRange) {
	throw new Error(`${codingAgent.manifestPath} does not depend on ${agentCore.upstreamName}`);
}
// The alias keeps every import of the upstream name working without a source rename.
codingAgent.manifest.dependencies[agentCore.upstreamName] =
	`npm:${agentCore.name}@${agentCore.version}`;

for (const pkg of packages) {
	pkg.manifest.name = pkg.name;
	pkg.manifest.version = pkg.version;
	// npm provenance rejects a package whose repository differs from the one that built it.
	pkg.manifest.repository = { type: "git", url: REPOSITORY_URL, directory: `packages/${pkg.dir}` };
	pkg.manifest.publishConfig = { access: "public" };

	const readmePath = join(pkg.directory, "README.md");
	const readme = readFileSync(readmePath, "utf8");
	if (!check) {
		writeFileSync(pkg.manifestPath, `${JSON.stringify(pkg.manifest, null, "\t")}\n`);
		if (!readme.startsWith(README_MARKER)) writeFileSync(readmePath, README_NOTE + readme);
		// MIT requires the copyright notice in every copy. npm packs LICENSE from the package root.
		const packageLicense = join(pkg.directory, "LICENSE");
		if (!existsSync(packageLicense)) copyFileSync(license, packageLicense);
	}
	console.log(`${pkg.name}@${pkg.version}`);
}
if (check) console.log("Checked only, nothing was written.");
