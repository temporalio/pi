# Temporal's build of pi

This fork adds the step-level API that [pi-temporal](https://github.com/temporalio/pi-temporal)
needs. Only two packages differ from upstream at runtime, so only those two are published.

| Package | Published as |
| --- | --- |
| `packages/agent` | `@temporalio/pi-agent-core` |
| `packages/coding-agent` | `@temporalio/pi-coding-agent` |

All other pi packages (`@earendil-works/pi-ai`, `@earendil-works/pi-tui` and so on) come from
upstream's npm releases.

## Versions

A build has the version `<upstream version>-temporal.<pi-temporal version>`. For example,
`1.1.0-temporal.0.2.0` is upstream `1.1.0` with the fork commit that pi-temporal `0.2.0` pins.

The coding agent depends on the agent core through an npm alias.

```json
"@earendil-works/pi-agent-core": "npm:@temporalio/pi-agent-core@1.1.0-temporal.0.2.0"
```

The source keeps its upstream imports, and an install has one copy of the agent core. The CLI
bundles the agent core. The library entry imports it at run time through the alias.

The CLI is also called `pi`. Do not install it globally next to the upstream `pi`.

## Releases

Releases are published by hand, the same way `temporalio/sdk-typescript` publishes. No CI job
publishes to npm.

1. Merge the fork change into `main`.
2. Tag the commit that pi-temporal pins as `pi-temporal-vX.Y.Z`, where `X.Y.Z` is the pi-temporal
   version, and push the tag.
3. Wait for the `Temporal packages` run on the tag to pass. It rewrites and packs both packages,
   installs them in a fresh project and checks the step API and the `pi` bin. It publishes nothing.
4. A maintainer with publish rights on the `@temporalio` scope publishes from the tag. Log in to
   npm as `temporal-sdk-team`, the account that `sdk-typescript` uses.

```sh
npm login
git fetch origin --tags
git checkout pi-temporal-vX.Y.Z
git merge-base --is-ancestor HEAD origin/main
npm ci --ignore-scripts
npm run build
node scripts/temporal-publish.mjs X.Y.Z
(cd packages/agent && npm publish --access public --tag latest --ignore-scripts)
(cd packages/coding-agent && npm publish --access public --tag latest --ignore-scripts)
git checkout -- packages
rm packages/agent/LICENSE packages/coding-agent/LICENSE
```

Publish the agent core first, because the coding agent depends on its exact version. The
versions are prereleases, so `npm publish` needs the explicit `--tag latest`. `--ignore-scripts`
stops `prepublishOnly` from cleaning and building the coding agent again. Add `--dry-run` to both
`npm publish` commands to see what would go out.

`node scripts/temporal-publish.mjs X.Y.Z --check` validates without writing.

## One-time npm setup

The Temporal SDK team creates `@temporalio/pi-agent-core` and `@temporalio/pi-coding-agent` on
the `@temporalio` scope, with publish rights for `temporal-sdk-team`. The first manual publish
creates them if the account can publish new packages on the scope.
