import fs from 'node:fs'
import path from 'node:path'

const SERIES_DIR = path.resolve(import.meta.dirname, '..', 'series')

// 文脈を読まずに誤答だと判断できる断定表現。正解には適用しない。
const cuePatterns = [
    /必ず/,
    /完全に/,
    /すべての/,
    /常に/,
    /絶対に/,
    /自動的に/,
    /無条件に/,
    /保証する/,
    /保証さ/,
    /証明された/,
    /一切/,
    /まったく/,
]

function quizFiles(dir) {
    return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
        const file = path.join(dir, entry.name)
        if (entry.isDirectory()) return quizFiles(file)
        return entry.name === 'quiz.md' ? [file] : []
    })
}

// 選択肢直後の独立行 <!-- quiz-lint-ignore: 種別[,種別...] - 理由 --> で、
// 指定した種別の警告（機械検出の誤検知）だけをその選択肢に限定して抑制できる。
// 種別: length（長さ非対称）, hedge（構文非対称）
const ignoreCommentPattern = /^<!--\s*quiz-lint-ignore:\s*([a-z,]+)\s*-\s*(.+?)\s*-->$/
const knownIgnoreKinds = new Set(['length', 'hedge'])

function parseQuizzes(file) {
    const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/)
    const quizzes = []
    let question = null

    for (let i = 0; i < lines.length; i++) {
        const heading = lines[i].match(/^### Q\d+\.\s*(.+)$/)
        if (heading) {
            question = { heading: heading[1], line: i + 1, choices: [], answer: -1 }
            continue
        }
        if (!question || lines[i].trim() !== '```quiz') continue

        for (i++; i < lines.length && lines[i].trim() !== '```'; i++) {
            const line = lines[i].trim()
            if (!line.startsWith('- ')) continue
            let choice = line.slice(2).trim()
            if (choice.startsWith('[x]')) {
                question.answer = question.choices.length
                choice = choice.slice(3).trim()
            }
            let lookahead = i + 1
            if ((lines[lookahead]?.trim() ?? '').startsWith('R:')) lookahead++
            const next = lines[lookahead]?.trim() ?? ''
            const ignoreMatch = next.match(ignoreCommentPattern)
            const ignoreKinds = new Set(
                (ignoreMatch?.[1] ?? '')
                    .split(',')
                    .map((kind) => kind.trim())
                    .filter((kind) => knownIgnoreKinds.has(kind)),
            )
            question.choices.push({ text: choice, line: i + 1, ignoreKinds })
        }
        quizzes.push(question)
        question = null
    }
    return quizzes
}

const errors = []
const warnings = []
const summaries = []

for (const file of quizFiles(SERIES_DIR).sort()) {
    const relative = path.relative(path.resolve(SERIES_DIR, '..', '..'), file)
    const quizzes = parseQuizzes(file)
    const positions = []
    const eligiblePositions = []

    for (const quiz of quizzes) {
        for (let index = 0; index < quiz.choices.length; index++) {
            eligiblePositions[index] = (eligiblePositions[index] ?? 0) + 1
            positions[index] = positions[index] ?? 0
        }
        if (quiz.answer >= 0) positions[quiz.answer] = (positions[quiz.answer] ?? 0) + 1
        const seen = new Set()
        for (const choice of quiz.choices) {
            if (seen.has(choice.text)) {
                errors.push(`${relative}:${choice.line}: 選択肢が重複しています`)
            }
            seen.add(choice.text)
        }
        for (const [index, choice] of quiz.choices.entries()) {
            if (index === quiz.answer) continue
            if (cuePatterns.some((pattern) => pattern.test(choice.text))) {
                errors.push(`${relative}:${choice.line}: 誤答に断定語の手がかりがあります: ${choice.text}`)
            }
        }

        const answerChoice = quiz.answer >= 0 ? quiz.choices[quiz.answer] : null

        // 長さ非対称: 正解だけが最長（かつ他より大幅に長い）だと、内容を読まずに選べてしまう。
        if (answerChoice && !answerChoice.ignoreKinds.has('length')) {
            const lengths = quiz.choices.map((choice) => choice.text.length)
            const answerLength = lengths[quiz.answer]
            const others = lengths.filter((_, index) => index !== quiz.answer)
            const maxOther = Math.max(...others)
            const avgOther = others.reduce((sum, length) => sum + length, 0) / others.length
            if (answerLength === Math.max(...lengths) && answerLength > maxOther && answerLength >= avgOther * 1.5) {
                warnings.push(
                    `${relative}:${answerChoice.line}: 「${quiz.heading}」の正解が他より大幅に長い可能性があります（正解${answerLength}字 / 誤答平均${avgOther.toFixed(1)}字）`,
                )
            }
        }

        // 構文非対称: 正解だけが「〜が（一方）〜とは限らない／依存する」型の条件付き構文で、
        // 誤答が単純断定のままだと、歯切れの良さだけで正解が推測できてしまう。
        const hedgePattern = /(とは限らない|に依存する|場合がある|とは言えない|とは異なる|わけではない)/
        if (answerChoice && !answerChoice.ignoreKinds.has('hedge')) {
            const othersHedged = quiz.choices.some((choice, index) => index !== quiz.answer && hedgePattern.test(choice.text))
            if (hedgePattern.test(answerChoice.text) && !othersHedged) {
                warnings.push(
                    `${relative}:${answerChoice.line}: 「${quiz.heading}」正解のみが条件付き構文（${answerChoice.text.match(hedgePattern)?.[0]}）の可能性があります`,
                )
            }
        }
    }

    // 正解位置の完全均等は要求せず、同じ位置に極端に偏った場合だけ警告する。
    // 選択肢数が混在する場合も、その位置が存在する問題だけを分母にする。
    const biasedPositions = positions.flatMap((count, index) => {
        const eligible = eligiblePositions[index] ?? 0
        if (eligible < 4 || count / eligible < 0.75) return []
        return [`${index + 1}番目（${count}/${eligible}問）`]
    })
    if (biasedPositions.length > 0) {
        warnings.push(`${relative}: 正解位置が極端に偏っています（${biasedPositions.join('、')}）`)
    }
    summaries.push(`${relative}: ${quizzes.length}問, 正解位置 ${positions.map((count) => count ?? 0).join('/')}`)
}

if (errors.length > 0) {
    console.error(errors.join('\n'))
    process.exitCode = 1
}

console.log(`quiz品質チェック: ${summaries.length}ファイル, ${summaries.reduce((sum, line) => sum + Number(line.match(/: (\d+)問/)?.[1] ?? 0), 0)}問`)
for (const summary of summaries) console.log(`  ${summary}`)
for (const warning of warnings) console.warn(`⚠️ ${warning}`)
