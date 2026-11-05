// Build the separately authored Lovable frontend into a Pages-only artifact.
// Requires Node 22+ and a committed npm lockfile. No deployment or credentials.
import { appendFileSync, cpSync, existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

export function inside(parent, candidate) {
  const path = relative(parent, candidate);
  return path !== '' && !path.startsWith(`..${sep}`) && path !== '..' && !isAbsolute(path);
}

export function readConfig(root = ROOT) {
  const config = JSON.parse(readFileSync(join(root, '.github', 'website.json'), 'utf8'));
  const app = resolve(root, config.directory);
  const output = resolve(app, config.output);
  if (!inside(root, app) || !inside(app, output)) {
    throw new Error('Website source and output must be separate directories inside the checkout.');
  }
  const url = new URL(config.url);
  if (url.protocol !== 'https:' || url.pathname !== '/' || url.username || url.password || url.port || url.search || url.hash) {
    throw new Error('Website URL must be an HTTPS origin without a path or credentials.');
  }
  if (existsSync(app) && !inside(realpathSync(root), realpathSync(app))) {
    throw new Error('Website source resolves outside the checkout.');
  }
  return { ...config, app, output, url: url.origin };
}

export function validateArtifact(directory) {
  if (!existsSync(join(directory, 'index.html'))) {
    throw new Error('No index.html in the build output. GitHub Pages requires a static export; a server-only Lovable/TanStack build cannot be deployed here.');
  }
  let bytes = 0;
  const visit = (path) => {
    const stat = lstatSync(path);
    if (stat.isSymbolicLink() || (!stat.isDirectory() && stat.nlink > 1)) {
      throw new Error(`Pages artifact cannot contain symbolic or hard links: ${path}`);
    }
    if (stat.isDirectory()) {
      for (const entry of readdirSync(path)) {
        if (entry === '.git' || entry === 'node_modules' || entry === 'package.json' || entry.startsWith('.env') || entry.endsWith('.map')) {
          throw new Error(`Unexpected source, dependency, environment or source-map file in the Pages artifact: ${entry}`);
        }
        visit(join(path, entry));
      }
    } else bytes += stat.size;
  };
  visit(directory);
  if (bytes > 1_000_000_000) throw new Error('Website artifact exceeds the GitHub Pages 1 GB limit.');
  const html = readFileSync(join(directory, 'index.html'), 'utf8');
  if (!/<html[\s>]/i.test(html) || !/<title[\s>]/i.test(html)) {
    throw new Error('Static entry point must be an HTML document with a title.');
  }
  // Only inspect markup, not HTML examples inside scripts, styles or comments.
  const markup = html.replace(/<!--[\s\S]*?-->/g, '')
    .replace(/<(script|style)\b[^>]*>[\s\S]*?<\/\1\s*>/gi, '');
  const ids = new Set();
  for (const [, id] of markup.matchAll(/<[^>]*?\s+id\s*=\s*["']([^"']+)["'][^>]*>/gi)) {
    if (ids.has(id)) throw new Error(`Duplicate HTML id in website entry point: ${id}`);
    ids.add(id);
  }
  if (/(?:src|href)=["']\/AgentQuantix\//i.test(html)) {
    throw new Error('Build assets target /AgentQuantix/. The custom domain requires base /.');
  }
  for (const [, value] of html.matchAll(/(?:src|href)=["']([^"']+)["']/gi)) {
    if (/^(?:https?:|data:|mailto:|#|\/\/)/i.test(value)) continue;
    const file = decodeURIComponent(value.split(/[?#]/)[0]).replace(/^\//, '');
    if (!file) continue;
    const asset = resolve(directory, file);
    if (!inside(directory, asset) || !existsSync(asset)) {
      throw new Error(`Broken or unsafe entry-point asset: ${value}`);
    }
  }
  return bytes;
}

function githubOutput(ready) {
  if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, `ready=${ready}\n`);
}

function npm(args, cwd) {
  // Arguments are fixed here, never interpolated from website configuration.
  const result = spawnSync(process.platform === 'win32' ? 'npm.cmd' : 'npm', args,
    { cwd, stdio: 'inherit', shell: process.platform === 'win32' });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`npm ${args[0]} failed (${result.status}).`);
}

export function main(root = ROOT) {
  const config = readConfig(root);
  const staging = join(root, 'build', 'website');
  if (!existsSync(join(config.app, 'package.json'))) {
    if (config.require_source) throw new Error('Required Lovable source is missing from website/.');
    console.log('Waiting for the Lovable export in website/. Configuration checked; no artifact or deployment was produced.');
    githubOutput(false);
    return;
  }
  if (!existsSync(join(config.app, 'package-lock.json')) && !existsSync(join(config.app, 'npm-shrinkwrap.json'))) {
    throw new Error('Commit an npm lockfile with the Lovable export. Generate it with npm install --package-lock-only in website/.');
  }
  const pkg = JSON.parse(readFileSync(join(config.app, 'package.json'), 'utf8'));
  if (!pkg.scripts?.build) throw new Error('The Lovable export must define an npm build script.');
  npm(['ci', '--no-audit', '--no-fund'], config.app);
  npm(['run', 'build', '--', '--base=/'], config.app);
  if (!inside(realpathSync(config.app), realpathSync(config.output))) {
    throw new Error('Website output resolves outside its source directory.');
  }
  const bytes = validateArtifact(config.output);
  // Validate the fixed absolute staging target before replacing generated output.
  const expectedParent = join(realpathSync(root), 'build');
  mkdirSync(dirname(staging), { recursive: true });
  if (realpathSync(dirname(staging)) !== expectedParent || (existsSync(staging) && lstatSync(staging).isSymbolicLink())) {
    throw new Error('Refusing to replace an unexpected staging directory.');
  }
  rmSync(staging, { recursive: true, force: true });
  cpSync(config.output, staging, { recursive: true });
  writeFileSync(join(staging, '.nojekyll'), '');
  console.log(`Validated ${bytes} bytes for ${config.url}; Pages artifact: ${staging}`);
  githubOutput(true);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) { console.error(`Website build failed: ${error.message}`); process.exitCode = 1; }
}
