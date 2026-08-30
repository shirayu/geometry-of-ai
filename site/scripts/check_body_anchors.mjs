import fs from 'node:fs'
import path from 'node:path'

const SITE_DIR = path.resolve(import.meta.dirname, '..')
const SERIES_DIR = path.join(SITE_DIR, 'series')

export function slugifyHeading(text) {
    const rControl = /[\u0000-\u001f]/g
    const rSpecial = /[\s~`!@#$%^&*()\-_+=[\]{}|\\;:"'“”‘’<>,.?/]+/g
    const rCombining = /[\u0300-\u036F]/g
    return text
        .normalize('NFKD')
        .replace(rCombining, '')
        // NFKDは濁点・半濁点付き仮名を基底文字+結合濁点・半濁点（U+3099/U+309A）に分解するが、
        // これらはrCombiningの範囲（U+0300-U+036F）外のため、ここでNFCにより合成済み仮名へ戻す。
        // ラテン文字のアクセント（U+0300-U+036F）は既に除去済みなので再合成されない（é→eのまま）。
        .normalize('NFC')
        .replace(rControl, '')
        .replace(rSpecial, '-')
        .replace(/-{2,}/g, '-')
        .replace(/^-+|-+$/g, '')
        .replace(/^(\d)/, '_$1')
        .toLowerCase()
}

export function headingSlugs(file) {
    const slugs = new Set()
    const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/)
    let inFence = false
    const anchorIdRe = /<a\s[^>]*\bid="([^"]+)"[^>]*>/g
    for (const line of lines) {
        if (/^\s*```/.test(line)) {
            inFence = !inFence
            continue
        }
        if (inFence) continue
        const heading = line.match(/^#{2,3} (.+)$/)
        if (heading) slugs.add(slugifyHeading(heading[1].trim()))
        let anchorMatch = null
        while ((anchorMatch = anchorIdRe.exec(line)) !== null) {
            slugs.add(anchorMatch[1])
        }
    }
    return slugs
}

export function bodyFiles(dir) {
    return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
        const file = path.join(dir, entry.name)
        if (entry.isDirectory()) return []
        return entry.name.endsWith('.md') ? [file] : []
    })
}

// 本文中の同一ファイル内アンカーリンク（例: [統計学](#統計学-...)）が、
// 実際にその見出し・<a id>アンカーを持つかを検証する。
// markdownlint MD051 はGitHub基準のslug生成で検証するが、本サイトはVitePress基準の
// slug生成でビルドされ、日本語見出しでは両者の結果が一致しないため、MD051は無効化し
// このスクリプトで代替検証する。
export function checkBodyAnchorLinks(file, errors) {
    const relative = path.relative(path.resolve(SITE_DIR, '..'), file)
    const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/)
    const slugs = headingSlugs(file)
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
            if (!slugs.has(fragment)) {
                errors.push(`${relative}:${i + 1}: 本文内アンカーリンクのslugが存在しません: #${fragment}`)
            }
        }
    }
}

async function main() {
    const errors = []
    for (const file of bodyFiles(SERIES_DIR)) {
        checkBodyAnchorLinks(file, errors)
    }

    if (errors.length > 0) {
        console.error(errors.map((error) => `❌ ${error}`).join('\n'))
        process.exit(1)
    }

    console.log(`✅ 本文内アンカーリンクチェック完了（${bodyFiles(SERIES_DIR).length}ファイル）`)
}

if (import.meta.url === `file://${process.argv[1]}`) {
    await main()
}
