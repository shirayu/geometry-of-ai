import { defineConfig } from 'vitepress'
import type MarkdownIt from 'markdown-it'

// ```filename.py → ```python [filename.py]
function transformPythonFences(md: MarkdownIt) {
    const original = md.renderer.rules.fence ?? ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options))
    md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        const info = token.info.trim()
        if (/^[a-zA-Z0-9_]+\.py$/.test(info)) {
            token.info = `python [${info}]`
        }
        return original(tokens, idx, options, env, self)
    }
}

// ```mermaid → MermaidDiagram コンポーネントでクライアント描画
function transformMermaidFences(md: MarkdownIt) {
    const original = md.renderer.rules.fence ?? ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options))
    md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        const info = token.info.trim()
        if (info === 'mermaid') {
            const codeBase64 = Buffer.from(token.content, 'utf8').toString('base64')
            return `<MermaidDiagram code-base64="${codeBase64}" />`
        }
        return original(tokens, idx, options, env, self)
    }
}

// ```quiz → Quiz コンポーネントでクリック即正誤判定
// 問題文・レベルは Markdown の見出し（###）側で書く。フェンスは選択肢と解説のみ。
// フォーマット:
//   - 選択肢1
//   - [x]選択肢2  (正解)
//   - 選択肢3
//   A: 解説文
//   S: /series/02#見出しslug
function transformQuizFences(md: MarkdownIt) {
    const original = md.renderer.rules.fence ?? ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options))
    md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        const info = token.info.trim()
        if (info === 'quiz') {
            const lines = token.content.split('\n')
            let answer = -1
            const choices: string[] = []
            let explanation = ''
            const sources: string[] = []
            for (const raw of lines) {
                const line = raw.trim()
                if (!line) continue
                if (line.startsWith('A:')) {
                    if (explanation) {
                        throw new Error(`quizの解説が複数あります: ${env.relativePath ?? 'unknown'}`)
                    }
                    explanation = md.renderInline(line.slice('A:'.length).trim()).trim()
                } else if (line.startsWith('S:')) {
                    const source = line.slice('S:'.length).trim()
                    if (!/^\/series\/[^\s#]+#[^\s#]+$/.test(source)) {
                        throw new Error(`quizの本文リンクは /series/ファイル#見出しslug 形式にしてください: ${env.relativePath ?? 'unknown'}`)
                    }
                    if (!sources.includes(source)) sources.push(source)
                } else if (line.startsWith('-')) {
                    const body = line.slice(1).trim()
                    const isCorrect = body.startsWith('[x]')
                    if (isCorrect) {
                        if (answer !== -1) {
                            throw new Error(`quizの正解は1つだけ指定してください: ${env.relativePath ?? 'unknown'}`)
                        }
                        answer = choices.length
                    }
                    const choice = isCorrect ? body.slice(3).trim() : body
                    if (!choice) {
                        throw new Error(`quizの選択肢が空です: ${env.relativePath ?? 'unknown'}`)
                    }
                    choices.push(md.renderInline(choice).trim())
                }
            }
            if (choices.length < 3 || choices.length > 5) {
                throw new Error(`quizの選択肢は3〜5個にしてください: ${env.relativePath ?? 'unknown'}`)
            }
            if (answer === -1) {
                throw new Error(`quizの正解を[x]で1つ指定してください: ${env.relativePath ?? 'unknown'}`)
            }
            if (!explanation) {
                throw new Error(`quizの解説がありません: ${env.relativePath ?? 'unknown'}`)
            }
            if (sources.length === 0) {
                throw new Error(`quizの本文リンクS:が少なくとも1つ必要です: ${env.relativePath ?? 'unknown'}`)
            }
            const data = { choices, answer, explanation, sources }
            const dataBase64 = Buffer.from(JSON.stringify(data), 'utf8').toString('base64')
            return `<Quiz data-base64="${dataBase64}" />`
        }
        return original(tokens, idx, options, env, self)
    }
}

// <details>\n<summary>label</summary> → <details v-pre> で Vue 補間を無効化
function transformDetails(md: MarkdownIt) {
    md.core.ruler.push('details_to_vitepress', (state) => {
        const tokens = state.tokens
        for (let i = 0; i < tokens.length; i++) {
            const t = tokens[i]
            if (t.type !== 'html_block') continue
            t.content = t.content
                .replace(/<details>\n<summary>/g, '<details v-pre>\n<summary>')
        }
    })
}

// src="filename.ext" (相対パス) → src="/series/filename.ext"
function transformImagePaths(md: MarkdownIt) {
    md.core.ruler.push('image_paths', (state) => {
        for (const t of state.tokens) {
            if (t.type === 'html_block') {
                t.content = t.content.replace(
                    /src="(?!\/|https?:\/\/)([^"]+\.(svg|png|jpg|jpeg|gif|webp))"/gi,
                    'src="/series/$1"',
                )
            }
        }
    })
}

// keywords.md / references.md に aside: false frontmatter を付与
function injectFrontmatter(md: MarkdownIt) {
    md.core.ruler.push('inject_frontmatter', (state) => {
        const file = (state.env as { relativePath?: string }).relativePath ?? ''
        if (file !== 'series/keywords.md' && file !== 'series/references.md' && file !== 'series/quizzes.md') return
        if (state.env && !(state.env as Record<string, unknown>).frontmatter) {
            ;(state.env as Record<string, unknown>).frontmatter = { aside: false }
        }
    })
}

export default defineConfig({
    title: '情報幾何学とAI',
    description: 'AIの表現空間設計を「幾何学」という言語で読み解く全15回の講義',
    lang: 'ja',

    head: [
        ['link', { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }],
    ],

    srcDir: '.',
    outDir: '.vitepress/dist',
    cleanUrls: true,
    sitemap: {
        hostname: 'https://geometry-of-ai.hayashibe.jp',
    },

    markdown: {
        math: true,
        config: (md) => {
            transformMermaidFences(md)
            transformQuizFences(md)
            transformPythonFences(md)
            transformDetails(md)
            transformImagePaths(md)
            injectFrontmatter(md)
        },
    },

    themeConfig: {
        nav: [],

        sidebar: [
            {
                text: '',
                items: [
                    { text: '重要な前提と制約', link: '/series/intro' },
                    { text: '全ページ目次', link: '/series/toc' },
                    { text: 'キーワード集', link: '/series/keywords' },
                    { text: '理解度チェック', link: '/series/quizzes' },
                    { text: '参考文献', link: '/series/references' },
                ],
            },
            {
                text: '第0部 準備と地図',
                items: [
                    {
                        text: '',
                        items: [
                            { text: '#0 幾何学という言語', link: '/series/00' },
                        ],
                    },
                ],
            },
            {
                text: '第1部 平坦な世界の限界',
                items: [
                    {
                        text: '',
                        items: [
                            { text: '#1 かつての地図', link: '/series/01' },
                            { text: '#2 ノルムの呪い', link: '/series/02' },
                            { text: '#3 プラネタリウムの建設', link: '/series/03' },
                        ],
                    },
                ],
            },
            {
                text: '第2部 統一的視点への接続',
                items: [
                    {
                        text: '',
                        items: [
                            { text: '#4 分類の再統一 I', link: '/series/04' },
                            { text: '#5 分類の再統一 II', link: '/series/05' },
                            { text: '#6 Transformerという測量士', link: '/series/06' },
                            { text: '#7 不確実性の復権', link: '/series/07' },
                        ],
                    },
                ],
            },
            {
                text: '第3部 時間とダイナミクス',
                items: [
                    {
                        text: '',
                        items: [
                            { text: '#8 時間の発見', link: '/series/08' },
                            { text: '#9 拡散と凝縮', link: '/series/09' },
                            { text: '#10 思考の連鎖', link: '/series/10' },
                        ],
                    },
                ],
            },
            {
                text: '第4部 マルチモーダルと拡張幾何学',
                items: [
                    {
                        text: '',
                        items: [
                            { text: '#11 感覚の統合', link: '/series/11' },
                            { text: '#12 双曲幾何学', link: '/series/12' },
                        ],
                    },
                ],
            },
            {
                text: '第5部 未来と哲学',
                items: [
                    {
                        text: '',
                        items: [
                            { text: '#13 高次元の深淵', link: '/series/13' },
                            { text: '#14 トポロジーという顕微鏡', link: '/series/14' },
                            { text: '#15 次の時代を設計する', link: '/series/15' },
                        ],
                    },
                ],
            },
            {
                text: 'Appendix',
                items: [
                    {
                        text: '',
                        items: [
                            { text: 'A1 量子化の幾何学', link: '/series/appendix.1' },
                            { text: 'A2 多様体の純度問題', link: '/series/appendix.2' },
                            { text: 'A3 動的剪定の幾何学', link: '/series/appendix.3' },
                            { text: 'A4 空間の「物差し」再考', link: '/series/appendix.4' },
                            { text: 'A5 情報幾何学における双対構造', link: '/series/appendix.5' },
                            { text: 'A6 特異点の幾何学', link: '/series/appendix.6' },
                        ],
                    },
                ],
            },
        ],

        socialLinks: [
            { icon: 'github', link: 'https://github.com/shirayu/geometry-of-ai' },
        ],

        search: {
            provider: 'local',
        },

        outline: {
            level: [2, 3],
            label: '目次',
        },

        docFooter: {
            prev: '前へ',
            next: '次へ',
        },
    },
})
