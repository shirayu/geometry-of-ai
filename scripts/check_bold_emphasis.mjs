import fs from 'node:fs';
import path from 'node:path';
import MarkdownIt from 'markdown-it';

const md = new MarkdownIt();

const getFiles = (dir) => {
    const files = fs.readdirSync(dir, {withFileTypes: true});
    return files.flatMap(file => {
        const res = path.resolve(dir, file.name);
        if (file.isDirectory()) {
            if (file.name === 'node_modules' || file.name === '.git') return [];
            return getFiles(res);
        }
        return file.name.endsWith('.md') ? res : [];
    });
};

const dirs = process.argv.slice(2);
if (dirs.length === 0) { console.error('Usage: check_bold_emphasis.mjs <dir>...'); process.exit(1); }
const files = dirs.flatMap(d => getFiles(d));
let hasError = false;

// **強調** / __強調__ が、直前直後の全角括弧・句読点などとの組み合わせにより
// CommonMark のフランキング規則で開始/終了デリミタとして成立せず、
// リテラルの ** がそのまま出力されてしまうケースを検出する。
// コードフェンス内は対象外とする（`**` はべき乗演算子として頻出する）。
files.forEach(file => {
    const lines = fs.readFileSync(file, 'utf-8').split('\n');
    let inFence = false;
    let fenceMarker = '';

    lines.forEach((line, i) => {
        const stripped = line.trim();
        const fenceMatch = stripped.match(/^(`{3,}|~{3,})/);
        if (fenceMatch) {
            if (!inFence) {
                inFence = true;
                fenceMarker = fenceMatch[1][0];
            } else if (stripped[0] === fenceMarker) {
                inFence = false;
                fenceMarker = '';
            }
            return;
        }
        if (inFence) return;
        if (!line.includes('**') && !line.includes('__')) return;

        const rendered = md.renderInline(line);
        if (rendered.includes('**') || rendered.includes('__')) {
            console.error(`${file}:${i + 1}: 強調記号が意図通りレンダリングされていません（前後の文字を確認）: ${line.trim()}`);
            hasError = true;
        }
    });
});

if (hasError) process.exit(1);
