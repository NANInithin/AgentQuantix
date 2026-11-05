# Lovable landing-page source

Place the exported landing-page app in this directory when it is ready. No page
is implemented here: design and content are being built in Lovable.

The Pages pipeline expects an npm `build` script, a committed `package-lock.json`
or `npm-shrinkwrap.json`, and a static `dist/index.html`. It runs `npm ci` and
`npm run build -- --base=/`. A server-only build is not compatible with Pages.

See [the deployment guide](../docs/WEBSITE.md) for branch rules, source import,
static-export requirements and the production custom domain.
