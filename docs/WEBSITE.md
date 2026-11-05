# Landing-page deployment

Production URL: **https://agentquantix.naninithin.me**.
Design and content are being built in Lovable. This change supplies deployment
infrastructure only; `website/README.md` is the source handoff contract.

## Source handoff from Lovable

1. Download/export the page's code or use its GitHub-synced repository.
2. Copy the frontend app into `website/` on `website/landing-page`. Keep the
   Python product's `src/`, packaging, workflows and Git history intact.
3. Include `package.json` with a `build` script and an npm lockfile. If the
   export only has a Bun lockfile, run `npm install --package-lock-only` in
   `website/` and commit the generated `package-lock.json`.
4. Ensure the build produces a static `website/dist/index.html` and its assets,
   with Vite base `/`. The workflow supplies `--base=/` to the build command.
5. Set `require_source` to `true` in `.github/website.json` for the first live
   website PR, so a later accidental source removal fails rather than skipping.
6. Open a PR into `main`. Review the Lovable preview and download the Actions
   artifact for a production-build check before merging.

Lovable's current GitHub integration creates its own repository when connecting
a project. Do not assume it can import this existing Python repository. Its
active sync branch should be a development branch, with PRs for production.
An automatic cross-repository import would need the actual Lovable repository
and a separately reviewed sync workflow; none is configured here.

### Static hosting compatibility

Older Lovable React/Vite apps build to static files. Newer Lovable apps use
TanStack Start and can require server-side rendering, server functions or API
routes. GitHub Pages cannot run that server. Ask Lovable for a static landing
page/export, or configure complete prerendering without runtime server features.
If its static output directory differs, update `output` in `.github/website.json`
and confirm the build command accepts the supplied Vite `--base` option.

The pipeline rejects builds without an HTML entry point. This is an artifact
check, not proof that a prerendered application has no server dependencies.
Review navigation, forms and integrations in the actual Lovable export before
the first production deployment. Pages does not provide SPA fallback rewrites;
use a one-page landing page, hash routing, or prerender each route.

## Branch and deployment configuration

- **`main`:** protected shared product/website branch and only production Pages
  deployment source. Website deployment is independent of package/model publication.
- **`website/landing-page`:** initial website feature branch; builds a review
  artifact when source is available. It never deploys production.
- **Future branches:** use short-lived branches and PRs into `main`.
- **PRs:** `website build` always reports, including on product-only PRs. There
  are no PR path filters that could leave a required check pending forever.
- **Production pushes:** only website/config/build-workflow changes redeploy.
  Product-only pushes do not. Manual dispatch supports a main redeployment.
- **Permissions:** build jobs have read-only repository access. Only the
  main-gated deployment job receives Pages/OIDC write permissions.
- **Environment:** `github-pages` must allow only the **branch** `main`.
- **Artifacts:** only validated static output is copied to `build/website` and
  uploaded. Symlinks, hard links, `.env*`, dependency directories and source maps
  are rejected. Dependencies are installed from the committed npm lockfile.
- **Actions:** dependencies in the new workflow are pinned to verified commit SHAs.

Until the Lovable code arrives, the job validates its configuration, logs
"Waiting for the Lovable export", and reports `ready=false`. Artifact upload
and deployment are skipped; no placeholder site is published.

GitHub Pages has one production site. The `website-preview` Actions artifact is
a downloadable build, not a hosted PR preview URL. Use Lovable's preview for
visual review and serve the downloaded build locally to check production assets.

## Local checks

Node 22+ and npm are required. No npm packages are needed before the Lovable
source is imported.

```powershell
node --test tests/website_pipeline.test.mjs
node scripts/build_website.mjs
```

Once a real export is present, the second command runs the locked install,
production build and artifact validation. To preview the result:

```powershell
python -m http.server 4173 --bind 127.0.0.1 --directory build/website
```

## One-time GitHub settings

After `website build` has appeared on the PR, use an authenticated administrator
GitHub CLI to apply the repository settings:

```powershell
python scripts/configure_github_website.py
python scripts/configure_github_website.py --apply
python scripts/configure_github_website.py --verify
```

The default prints a plan. Applying it requires PRs, all nine existing CI checks
plus `website build`, up-to-date branches and resolved review conversations;
blocks force pushes/deletion; and enforces protection for administrators.
Existing stricter review counts, code-owner rules, linear history and push
restrictions are preserved. It enables Actions-based Pages publishing and
restricts the `github-pages` environment to `main`, preserving existing required
environment reviewers and wait timers. It reads back and verifies the result.

UI equivalents:

- **Settings → Pages → Build and deployment → Source:** GitHub Actions.
- **Settings → Environments → github-pages → Deployment branches and tags:**
  Selected branches and tags, with the **branch** `main` only.
- **Settings → Branches → main rule:** add `website build` to the required
  checks alongside the existing CI checks, with GitHub Actions as the source.

The owner handles **DNS**, **Custom domain** and **Enforce HTTPS**. The setup
script does not modify these. Configure the custom domain in GitHub before
pointing DNS at it. A CNAME file does not configure a custom Actions deployment.

## Changing domains or rolling back

For a future domain move, change `url` in `.github/website.json`, the Lovable
page's canonical/Open Graph URLs, the Pages custom domain and DNS. Assets remain
at base `/`. Arrange redirects from the old address if needed.

Roll back website changes with a revert PR and let `main` redeploy. Do not
force-push `main`, deploy a feature branch, or move product release tags.

References: [Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages),
[publishing and deployment protection](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site),
[Lovable Git sync](https://docs.lovable.dev/integrations/github),
[Lovable framework and hosting compatibility](https://docs.lovable.dev/tips-tricks/deployment-hosting-ownership).
