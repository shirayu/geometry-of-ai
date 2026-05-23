
import fs from 'node:fs';
import path from 'node:path';
import katex from 'katex';

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

// 行番号を計算する関数
const getLineNumber = (content, index) => {
    return content.substring(0, index).split('\n').length;
};

const mathRegex = /\$\$([\s\S]+?)\$\$|\$([^\$]+?)\$/g;
const dirs = process.argv.slice(2);
if (dirs.length === 0) { console.error('Usage: check-math.mjs <dir>...'); process.exit(1); }
const files = dirs.flatMap(d => getFiles(d));
let hasError = false;

files.forEach(file => {
    const content = fs.readFileSync(file, 'utf-8');
    let match;

    while ((match = mathRegex.exec(content)) !== null) {
        const formula = (match[1] || match[2]).trim();
        const lineNum = getLineNumber(content, match.index);

        try {
            katex.renderToString(formula, {
                displayMode: !!match[1],
                throwOnError: true,
                strict: "ignore" // 警告（改行エラーなど）は無視して、致命的な構文エラーだけ出す
            });
        } catch (e) {
            const relativePath = path.relative(process.cwd(), file);
            // 行番号付きで出力
            console.error(`❌ Math Error in ${relativePath}:${lineNum}`);
            console.error(`   Formula: ${formula}`);
            console.error(`   Message: ${e.message}\n`);
            hasError = true;
        }
    }
});

if (hasError) process.exit(1);
console.log('✅ All math formulas are valid!');
