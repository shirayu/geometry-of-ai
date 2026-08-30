import fs from 'node:fs'
import path from 'node:path'
import { bodyFiles } from './check_body_anchors.mjs'

const SITE_DIR = path.resolve(import.meta.dirname, '..')
const SERIES_DIR = path.join(SITE_DIR, 'series')
const DIST_SERIES_DIR = path.join(SITE_DIR, '.vitepress', 'dist', 'series')

// check_body_anchors.mjs のslugify近似はインライン数式（$...$）を含む見出しで
// VitePressの実際のMathJaxレンダリング結果とずれることがある（例: 数式部分が
// 除去されたり、記号の一部だけが残ったりする）。このスクリプトは近似計算をせず、
// `task site:build` が生成した実際のHTMLの id 属性を正として、本文中の同一
// ファイル内アンカーリンク（[text](#slug)）が実在するかを機械的に検証する。
function htmlIds(file) {
    const ids = new Set()
    const html = fs.readFileSync(file, 'utf8')
    const idRe = /\sid="([^"]+)"/g
    let match = null
    while ((match = idRe.exec(html)) !== null) {
        ids.add(match[1])
    }
    return ids
}

function checkFile(mdFile, errors) {
    const relative = path.relative(path.resolve(SITE_DIR, '..'), mdFile)
    const base = path.basename(mdFile, '.md')
    const htmlFile = path.join(DIST_SERIES_DIR, `${base}.html`)

    if (!fs.existsSync(htmlFile)) {
        errors.push(`${relative}: 対応するビルド済みHTMLが見つかりません: ${path.relative(SITE_DIR, htmlFile)}（先に task site:build を実行してください）`)
        return
    }

    const ids = htmlIds(htmlFile)
    const lines = fs.readFileSync(mdFile, 'utf8').split(/\r?\n/)
    const linkRe = /\[[^\]]*\]\(#([^)]+)\)/g
    let inFence = false
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i]
        if (/^\s*```/.test(line)) {
            inFence = !inFence
            continue
        }
        if (inFence) continue
        let match = null
        while ((match = linkRe.exec(line)) !== null) {
            const fragment = match[1]
            if (fragment.startsWith('ref-')) continue
            if (!ids.has(fragment)) {
                errors.push(`${relative}:${i + 1}: ビルド後のHTMLに対応するidがありません: #${fragment}`)
            }
        }
    }
}

async function main() {
    if (!fs.existsSync(DIST_SERIES_DIR)) {
        console.error(`❌ ビルド済みディレクトリが見つかりません: ${DIST_SERIES_DIR}（先に task site:build を実行してください）`)
        process.exit(1)
    }

    const errors = []
    const files = bodyFiles(SERIES_DIR)
    for (const file of files) {
        checkFile(file, errors)
    }

    if (errors.length > 0) {
        console.error(errors.map((error) => `❌ ${error}`).join('\n'))
        process.exit(1)
    }

    console.log(`✅ ビルド後アンカーリンクチェック完了（${files.length}ファイル）`)
}

if (import.meta.url === `file://${process.argv[1]}`) {
    await main()
}
