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
}

const SERIES_DIR = path.resolve(__dirname, '.')

const PAGES: { file: string; link: string }[] = [
    { file: 'intro.md', link: '/series/intro' },
    { file: '00.md', link: '/series/00' },
    { file: '01.md', link: '/series/01' },
    { file: '02.md', link: '/series/02' },
    { file: '02/quiz.md', link: '/series/02/quiz' },
    { file: '03.md', link: '/series/03' },
    { file: '04.md', link: '/series/04' },
    { file: '05.md', link: '/series/05' },
    { file: '06.md', link: '/series/06' },
    { file: '07.md', link: '/series/07' },
    { file: '08.md', link: '/series/08' },
    { file: '09.md', link: '/series/09' },
    { file: '10.md', link: '/series/10' },
    { file: '11.md', link: '/series/11' },
    { file: '12.md', link: '/series/12' },
    { file: '13.md', link: '/series/13' },
    { file: '14.md', link: '/series/14' },
    { file: '15.md', link: '/series/15' },
    { file: 'appendix.1.md', link: '/series/appendix.1' },
    { file: 'appendix.2.md', link: '/series/appendix.2' },
    { file: 'appendix.3.md', link: '/series/appendix.3' },
    { file: 'appendix.4.md', link: '/series/appendix.4' },
    { file: 'appendix.5.md', link: '/series/appendix.5' },
    { file: 'appendix.6.md', link: '/series/appendix.6' },
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
        return PAGES.map(({ file, link }) => {
            const content = fs.readFileSync(path.join(SERIES_DIR, file), 'utf-8')
            const { title, headings } = extractHeadings(content)
            const id = file.replace('.md', '').replace(/\//g, '-')
            return { file, link, id, title, headings }
        })
    },
}
