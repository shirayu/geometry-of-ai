import { defineConfig } from 'vitepress'

export default defineConfig({
    title: '情報幾何学とAIの統一視点',
    description: '深層学習の歴史で紐解く、超球面上の「プラネタリウム」構築論',
    lang: 'ja',

    srcDir: '.',
    outDir: '.vitepress/dist',


    markdown: {
        math: true,
    },

    vue: {
        template: {
            compilerOptions: {
                // 数式内の {{ }} が Vue interpolation と衝突するのを防ぐ
                delimiters: ['${', '}'],
            },
        },
    },

    themeConfig: {
        nav: [
            { text: 'ホーム', link: '/' },
            { text: 'はじめに', link: '/series/intro' },
            { text: 'キーワード集', link: '/series/keywords' },
            { text: '参考文献', link: '/series/references' },
        ],

        sidebar: [
            {
                text: 'イントロ',
                items: [
                    { text: '重要な前提と制約', link: '/series/intro' },
                    { text: 'キーワード集', link: '/series/keywords' },
                    { text: '参考文献', link: '/series/references' },
                ],
            },
            {
                text: '第0部：準備と地図',
                items: [
                    { text: '第0回：幾何学という言語', link: '/series/00' },
                ],
            },
            {
                text: '第1部：平坦な世界の限界',
                items: [
                    { text: '第1回：かつての地図', link: '/series/01' },
                    { text: '第2回：ノルムの呪い', link: '/series/02' },
                    { text: '第3回：プラネタリウムの建設', link: '/series/03' },
                ],
            },
            {
                text: '第2部：統一的視点への接続',
                items: [
                    { text: '第4回：分類の再統一 I', link: '/series/04' },
                    { text: '第5回：分類の再統一 II', link: '/series/05' },
                    { text: '第6回：Transformerという測量士', link: '/series/06' },
                    { text: '第7回：不確実性の復権', link: '/series/07' },
                ],
            },
            {
                text: '第3部：時間とダイナミクス',
                items: [
                    { text: '第8回：時間の発見', link: '/series/08' },
                    { text: '第9回：拡散と凝縮', link: '/series/09' },
                    { text: '第10回：思考の連鎖', link: '/series/10' },
                ],
            },
            {
                text: '第4部：マルチモーダルと拡張幾何学',
                items: [
                    { text: '第11回：感覚の統合', link: '/series/11' },
                    { text: '第12回：双曲幾何学', link: '/series/12' },
                ],
            },
            {
                text: '第5部：未来と哲学',
                items: [
                    { text: '第13回：高次元の深淵', link: '/series/13' },
                    { text: '第14回：トポロジーという顕微鏡', link: '/series/14' },
                    { text: '第15回：次の時代を設計する', link: '/series/15' },
                ],
            },
            {
                text: 'Appendix',
                collapsed: true,
                items: [
                    { text: 'A1：量子化の幾何学', link: '/series/appendix.1' },
                    { text: 'A2：多様体の純度問題', link: '/series/appendix.2' },
                    { text: 'A3：動的剪定の幾何学', link: '/series/appendix.3' },
                    { text: 'A4：空間の「物差し」再考', link: '/series/appendix.4' },
                    { text: 'A5：情報幾何学における双対構造', link: '/series/appendix.5' },
                    { text: 'A6：特異点の幾何学', link: '/series/appendix.6' },
                ],
            },
        ],

        socialLinks: [
            { icon: 'github', link: 'https://github.com/yuta-h/geometry-of-ai' },
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
