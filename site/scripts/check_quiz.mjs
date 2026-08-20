import fs from 'node:fs'
import path from 'node:path'
import { createMarkdownRenderer } from 'vitepress'

const SITE_DIR = path.resolve(import.meta.dirname, '..')
const SERIES_DIR = path.join(SITE_DIR, 'series')

const forbiddenReferences = [
    { pattern: /本文/, label: '本文への参照' },
    { pattern: /(?:表|図|節)を参照/, label: '本文内要素への参照' },
    { pattern: /前述|上記|以下/, label: '前後の本文への参照' },
    { pattern: /次回/, label: '次回への参照' },
    {
        pattern: /本講義|本回|本シリーズ|この(?:章|講義|回|付録|シリーズ)|最終(?:メッセージ|的なメッセージ)|第[0-9０-９]+回/,
        label: '本文・講義への参照',
    },
]

function quizFiles(dir) {
    return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
        const file = path.join(dir, entry.name)
        if (entry.isDirectory()) return quizFiles(file)
        return entry.name === 'quiz.md' ? [file] : []
    })
}

function hasMath(source) {
    return /\$\$?[\s\S]*?\$\$?/.test(source)
}

export function slugifyHeading(text) {
    const rControl = /[\u0000-\u001f]/g
    const rSpecial = /[\s~`!@#$%^&*()\-_+=[\]{}|\\;:"'“”‘’<>,.?/]+/g
    const rCombining = /[\u0300-\u036F]/g
    return text
        .normalize('NFKD')
        .replace(rCombining, '')
        .replace(rControl, '')
        .replace(rSpecial, '-')
        .replace(/-{2,}/g, '-')
        .replace(/^-+|-+$/g, '')
        .replace(/^(\d)/, '_$1')
        .toLowerCase()
}

function sourceFile(href) {
    const [pathname, fragment] = href.split('#')
    if (!pathname.startsWith('/series/') || !fragment) return null

    const relative = pathname.slice('/series/'.length)
    if (!relative || relative.includes('..')) return null
    const filename = relative.endsWith('.md') ? relative : `${relative}.md`
    const file = path.resolve(SERIES_DIR, filename)
    if (!file.startsWith(`${SERIES_DIR}${path.sep}`)) return null
    return { file, fragment }
}

function headingSlugs(file) {
    const slugs = new Set()
    const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/)
    let inFence = false
    for (const line of lines) {
        if (/^\s*```/.test(line)) {
            inFence = !inFence
            continue
        }
        if (inFence) continue
        const heading = line.match(/^#{2,3} (.+)$/)
        if (heading) slugs.add(slugifyHeading(heading[1].trim()))
    }
    return slugs
}

function checkSource(href, file, line, errors) {
    const source = sourceFile(href)
    if (!source) {
        errors.push(`${file}:${line}: 本文リンクの形式またはパスが不正です: ${href}`)
        return
    }
    if (!fs.existsSync(source.file)) {
        errors.push(`${file}:${line}: 本文ファイルが存在しません: ${href}`)
        return
    }
    if (!headingSlugs(source.file).has(source.fragment)) {
        errors.push(`${file}:${line}: 本文の見出しslugが存在しません: ${href}`)
    }
}

function checkReferences(text, file, line, errors) {
    for (const { pattern, label } of forbiddenReferences) {
        if (pattern.test(text)) {
            errors.push(`${file}:${line}: ${label}をquiz内で使わないでください`)
        }
    }
}

function checkChapterQuizLinks(errors) {
    const chapters = [
        ...Array.from({ length: 16 }, (_, i) => ({ id: String(i).padStart(2, '0'), file: `${String(i).padStart(2, '0')}.md` })),
        ...Array.from({ length: 6 }, (_, i) => ({ id: `appendix.${i + 1}`, file: `appendix.${i + 1}.md` })),
    ]

    for (const { id, file } of chapters) {
        const bodyFile = path.join(SERIES_DIR, file)
        const quizFile = path.join(SERIES_DIR, id, 'quiz.md')
        const relative = path.relative(path.resolve(SITE_DIR, '..'), bodyFile)
        const href = `./${id}/quiz`

        if (!fs.existsSync(quizFile)) {
            errors.push(`${relative}: 対応するクイズファイルがありません: ${quizFile}`)
        }
        if (!fs.existsSync(bodyFile)) continue

        const body = fs.readFileSync(bodyFile, 'utf8')
        if (!body.includes(`[理解度チェック](${href})`)) {
            errors.push(`${relative}: 本文から理解度チェックへのリンクがありません: ${href}`)
        }
    }
}

async function main() {
    const md = await createMarkdownRenderer(SERIES_DIR, { math: true })
    const errors = []

    checkChapterQuizLinks(errors)

    for (const file of quizFiles(SERIES_DIR)) {
        const relative = path.relative(path.resolve(SITE_DIR, '..'), file)
        const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/)
        let inQuiz = false
        let blockStart = 0
        let choices = 0
        let answers = 0
        let reasons = 0
        let explanation = false
        let sources = []
        let questionText = ''

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i]
            const lineNumber = i + 1

            const questionHeading = line.match(/^### Q\d+\.\s*(.+)$/)
            if (!inQuiz && questionHeading) {
                questionText = ''
                checkReferences(questionHeading[1], relative, lineNumber, errors)
                continue
            }

            if (!inQuiz && questionText !== '' && line.trim() === '') continue

            if (!inQuiz && line.trim() === '```quiz') {
                inQuiz = true
                blockStart = lineNumber
                choices = 0
                answers = 0
                reasons = 0
                explanation = false
                sources = []
                checkReferences(questionText, relative, blockStart - 1, errors)
                continue
            }

            if (inQuiz && line.trim() === '```') {
                if (choices < 3 || choices > 5) {
                    errors.push(`${relative}:${blockStart}: 選択肢は3〜5個にしてください`)
                }
                if (answers !== 1) {
                    errors.push(`${relative}:${blockStart}: 正解指定[x]は1つだけ必要です`)
                }
                if (reasons !== choices) {
                    errors.push(`${relative}:${blockStart}: 各選択肢にR:（理由）が必要です`)
                }
                if (!explanation) {
                    errors.push(`${relative}:${blockStart}: 解説A:が必要です`)
                }
                if (sources.length === 0) {
                    errors.push(`${relative}:${blockStart}: 本文リンクS:が少なくとも1つ必要です`)
                }
                inQuiz = false
                continue
            }

            if (inQuiz) {
                const trimmed = line.trim()
                let text = ''
                if (trimmed.startsWith('-')) {
                    if (reasons !== choices) {
                        errors.push(`${relative}:${lineNumber}: 選択肢の直後にR:（理由）が必要です`)
                    }
                    choices++
                    const body = trimmed.slice(1).trim()
                    if (body.startsWith('[x]')) answers++
                    text = body.replace(/^\[x\]/, '').trim()
                } else if (trimmed.startsWith('R:')) {
                    reasons++
                    text = trimmed.slice(2).trim()
                    if (reasons > choices) {
                        errors.push(`${relative}:${lineNumber}: R:は選択肢の直後に書いてください`)
                    }
                } else if (trimmed.startsWith('A:')) {
                    explanation = true
                    text = trimmed.slice(2).trim()
                } else if (trimmed.startsWith('S:')) {
                    const source = trimmed.slice(2).trim()
                    if (!sources.includes(source)) sources.push(source)
                    checkSource(source, relative, lineNumber, errors)
                }

                if (text) {
                    checkReferences(text, relative, lineNumber, errors)
                    if (hasMath(text)) {
                        const rendered = md.renderInline(text)
                        if (!rendered.includes('mjx-container')) {
                            errors.push(`${relative}:${lineNumber}: 数式をMathJaxでレンダリングできません`)
                        }
                    }
                }
                continue
            }

            if (questionText !== undefined && line.trim() && !line.startsWith('#')) {
                questionText += `${line.trim()}\n`
            }
        }

        if (inQuiz) {
            errors.push(`${relative}:${blockStart}: quizフェンスが閉じていません`)
        }
    }

    if (errors.length > 0) {
        console.error(errors.map((error) => `❌ ${error}`).join('\n'))
        process.exit(1)
    }

    console.log(`✅ quizチェック完了（${quizFiles(SERIES_DIR).length}ファイル）`)
}

if (import.meta.url === `file://${process.argv[1]}`) {
    await main()
}
