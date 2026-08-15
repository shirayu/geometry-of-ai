import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import TocToggle from './TocToggle.vue'
import SearchButton from './SearchButton.vue'
import TocAside from './TocAside.vue'
import MermaidDiagram from './MermaidDiagram.vue'
import Quiz from './Quiz.vue'
import NoH2Counter from './NoH2Counter.vue'
import './custom.css'

export default {
    extends: DefaultTheme,
    enhanceApp({ app }) {
        app.component('MermaidDiagram', MermaidDiagram)
        app.component('Quiz', Quiz)
    },
    Layout: () => {
        return h(DefaultTheme.Layout, null, {
            'nav-bar-content-before': () => [h(TocToggle), h(SearchButton)],
            'aside-outline-before': () => h(TocAside),
            'layout-top': () => h(NoH2Counter),
        })
    },
}
