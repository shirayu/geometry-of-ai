import fs from 'node:fs'
import path from 'node:path'

export interface Heading {
    level: 2 | 3
    text: string
    slug: string
}

export interface PageToc {
    file: string
    link: string
    id: string
    title: string
    headings: Heading[]
    quizLink?: string
    part?: string
}

const SERIES_DIR = path.resolve(__dirname, '.')

// サイドバー（config.mts）の部構成と同期させること
const PAGES: { file: string; link: string; quizFile?: string; quizLink?: string; part?: string; noHeadings?: boolean }[] = [
    { file: 'intro.md', link: '/series/intro', part: 'ガイド' },
    { file: 'keywords.md', link: '/series/keywords', noHeadings: true, part: 'ガイド' },
    { file: 'references.md', link: '/series/references', noHeadings: true, part: 'ガイド' },
    { file: '00.md', link: '/series/00', quizFile: '00/quiz.md', quizLink: '/series/00/quiz', part: '第0部 準備と地図' },
    { file: '01.md', link: '/series/01', quizFile: '01/quiz.md', quizLink: '/series/01/quiz', part: '第1部 平坦な世界の限界' },
    { file: '02.md', link: '/series/02', quizFile: '02/quiz.md', quizLink: '/series/02/quiz', part: '第1部 平坦な世界の限界' },
    { file: '03.md', link: '/series/03', quizFile: '03/quiz.md', quizLink: '/series/03/quiz', part: '第1部 平坦な世界の限界' },
    { file: '04.md', link: '/series/04', quizFile: '04/quiz.md', quizLink: '/series/04/quiz', part: '第2部 空間の見方を広げる' },
    { file: '05.md', link: '/series/05', quizFile: '05/quiz.md', quizLink: '/series/05/quiz', part: '第2部 空間の見方を広げる' },
    { file: '06.md', link: '/series/06', quizFile: '06/quiz.md', quizLink: '/series/06/quiz', part: '第2部 空間の見方を広げる' },
    { file: '07.md', link: '/series/07', quizFile: '07/quiz.md', quizLink: '/series/07/quiz', part: '第2部 空間の見方を広げる' },
    { file: '08.md', link: '/series/08', quizFile: '08/quiz.md', quizLink: '/series/08/quiz', part: '第3部 時間とダイナミクス' },
    { file: '09.md', link: '/series/09', quizFile: '09/quiz.md', quizLink: '/series/09/quiz', part: '第3部 時間とダイナミクス' },
    { file: '10.md', link: '/series/10', quizFile: '10/quiz.md', quizLink: '/series/10/quiz', part: '第3部 時間とダイナミクス' },
    { file: '11.md', link: '/series/11', quizFile: '11/quiz.md', quizLink: '/series/11/quiz', part: '第4部 マルチモーダルと拡張幾何学' },
    { file: '12.md', link: '/series/12', quizFile: '12/quiz.md', quizLink: '/series/12/quiz', part: '第4部 マルチモーダルと拡張幾何学' },
    { file: '13.md', link: '/series/13', quizFile: '13/quiz.md', quizLink: '/series/13/quiz', part: '第5部 未来と哲学' },
    { file: '14.md', link: '/series/14', quizFile: '14/quiz.md', quizLink: '/series/14/quiz', part: '第5部 未来と哲学' },
    { file: '15.md', link: '/series/15', quizFile: '15/quiz.md', quizLink: '/series/15/quiz', part: '第5部 未来と哲学' },
    { file: 'appendix.1.md', link: '/series/appendix.1', quizFile: 'appendix.1/quiz.md', quizLink: '/series/appendix.1/quiz', part: 'Appendix' },
    { file: 'appendix.2.md', link: '/series/appendix.2', quizFile: 'appendix.2/quiz.md', quizLink: '/series/appendix.2/quiz', part: 'Appendix' },
    { file: 'appendix.3.md', link: '/series/appendix.3', quizFile: 'appendix.3/quiz.md', quizLink: '/series/appendix.3/quiz', part: 'Appendix' },
    { file: 'appendix.4.md', link: '/series/appendix.4', quizFile: 'appendix.4/quiz.md', quizLink: '/series/appendix.4/quiz', part: 'Appendix' },
    { file: 'appendix.5.md', link: '/series/appendix.5', quizFile: 'appendix.5/quiz.md', quizLink: '/series/appendix.5/quiz', part: 'Appendix' },
    { file: 'appendix.6.md', link: '/series/appendix.6', quizFile: 'appendix.6/quiz.md', quizLink: '/series/appendix.6/quiz', part: 'Appendix' },
]

function toSlug(text: string): string {
    return text
        .toLowerCase()
        .replace(/\s+/g, '-')
        .replace(/[^\p{L}\p{N}\-]/gu, '')
}

function extractHeadings(content: string): { title: string; headings: Heading[] } {
    const lines = content.split('\n')
    let title = ''
    const headings: Heading[] = []
    let inFence = false

    for (const line of lines) {
        if (/^```/.test(line)) {
            inFence = !inFence
            continue
        }
        if (inFence) continue

        const h1 = line.match(/^# (.+)/)
        if (h1 && !title) {
            title = h1[1].trim()
            continue
        }
        const h2 = line.match(/^## (.+)/)
        if (h2) {
            const text = h2[1].trim()
            headings.push({ level: 2, text, slug: toSlug(text) })
            continue
        }
        const h3 = line.match(/^### (.+)/)
        if (h3) {
            const text = h3[1].trim()
            headings.push({ level: 3, text, slug: toSlug(text) })
        }
    }

    return { title, headings }
}

export default {
    load(): PageToc[] {
        return PAGES.map(({ file, link, quizLink, part, noHeadings }) => {
            const content = fs.readFileSync(path.join(SERIES_DIR, file), 'utf-8')
            const { title, headings } = extractHeadings(content)
            const id = file.replace('.md', '').replace(/\//g, '-')
            return { file, link, id, title, headings: noHeadings ? [] : headings, quizLink, part }
        })
    },
}
