import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { inside, main, readConfig, validateArtifact } from '../scripts/build_website.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');

function fixture(t) {
  const directory = mkdtempSync(join(root, '.test-website-'));
  t.after(() => {
    assert.ok(inside(root, directory));
    rmSync(directory, { recursive: true, force: true });
  });
  mkdirSync(join(directory, '.github'));
  mkdirSync(join(directory, 'website'));
  writeFileSync(join(directory, '.github/website.json'), JSON.stringify({
    directory: 'website', output: 'dist', url: 'https://agentquantix.naninithin.me', require_source: false,
  }));
  return directory;
}

function artifact(directory) {
  mkdirSync(join(directory, 'assets'), { recursive: true });
  writeFileSync(join(directory, 'index.html'), '<html><head><title>Example</title></head><body><script src="/assets/app.js"></script></body></html>');
  writeFileSync(join(directory, 'assets/app.js'), 'console.log("example")');
}

test('source-free setup does not produce an artifact', (t) => {
  const directory = fixture(t);
  main(directory);
  assert.equal(existsSync(join(directory, 'build/website')), false);
});

test('required source cannot silently disappear', (t) => {
  const directory = fixture(t);
  writeFileSync(join(directory, '.github/website.json'), JSON.stringify({
    directory: 'website', output: 'dist', url: 'https://example.com', require_source: true,
  }));
  assert.throws(() => main(directory), /Required Lovable source is missing/);
});

test('configuration cannot publish the source checkout', (t) => {
  const directory = fixture(t);
  writeFileSync(join(directory, '.github/website.json'), JSON.stringify({
    directory: 'website', output: '..', url: 'https://example.com',
  }));
  assert.throws(() => readConfig(directory), /separate directories/);
});

test('static output is required', (t) => {
  const directory = fixture(t);
  assert.throws(() => validateArtifact(join(directory, 'website')), /requires a static export/);
});

test('a complete static artifact is accepted', (t) => {
  const directory = join(fixture(t), 'website/dist');
  artifact(directory);
  assert.ok(validateArtifact(directory) > 0);
});

for (const name of ['.env.production', 'private.map', 'package.json']) {
  test(`rejects accidental publication of ${name}`, (t) => {
    const directory = join(fixture(t), 'website/dist');
    artifact(directory);
    writeFileSync(join(directory, name), 'not for publication');
    assert.throws(() => validateArtifact(directory), /Unexpected source/);
  });
}

test('broken entry-point assets fail before publication', (t) => {
  const directory = join(fixture(t), 'website/dist');
  artifact(directory);
  writeFileSync(join(directory, 'index.html'), '<html><title>Broken</title><script src="/missing.js"></script></html>');
  assert.throws(() => validateArtifact(directory), /Broken or unsafe/);
});

test('repository base paths fail on the custom domain', (t) => {
  const directory = join(fixture(t), 'website/dist');
  artifact(directory);
  writeFileSync(join(directory, 'index.html'), '<html><title>Wrong base</title><script src="/AgentQuantix/assets/app.js"></script></html>');
  assert.throws(() => validateArtifact(directory), /requires base/);
});

test('npm ci, build, validation and staging work together without publishing source', (t) => {
  const directory = fixture(t);
  const app = join(directory, 'website');
  writeFileSync(join(app, 'package.json'), JSON.stringify({
    name: 'pipeline-fixture', version: '1.0.0', private: true, scripts: { build: 'node build.cjs' },
  }));
  writeFileSync(join(app, 'package-lock.json'), JSON.stringify({
    name: 'pipeline-fixture', version: '1.0.0', lockfileVersion: 3, requires: true,
    packages: { '': { name: 'pipeline-fixture', version: '1.0.0' } },
  }));
  writeFileSync(join(app, 'build.cjs'), `
    const fs = require('node:fs');
    if (!process.argv.includes('--base=/')) throw new Error('Expected custom-domain base');
    fs.mkdirSync('dist/assets', { recursive: true });
    fs.writeFileSync('dist/index.html', '<html><title>Fixture</title><script src="/assets/app.js"></script></html>');
    fs.writeFileSync('dist/assets/app.js', 'console.log("fixture")');
  `);
  main(directory);
  const staging = join(directory, 'build/website');
  assert.ok(readFileSync(join(staging, 'index.html'), 'utf8').includes('Fixture'));
  assert.ok(existsSync(join(staging, '.nojekyll')));
  assert.equal(existsSync(join(staging, 'package.json')), false);
  assert.equal(existsSync(join(staging, 'build.cjs')), false);
});
