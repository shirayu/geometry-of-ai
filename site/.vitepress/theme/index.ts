import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import TocToggle from './TocToggle.vue'
import SearchButton from './SearchButton.vue'
import TocAside from './TocAside.vue'
import MermaidDiagram from './MermaidDiagram.vue'
import './custom.css'

export default {
    extends: DefaultTheme,
    enhanceApp({ app }) {
        app.component('MermaidDiagram', MermaidDiagram)
    },
    Layout: () => {
        return h(DefaultTheme.Layout, null, {
            'nav-bar-content-before': () => [h(TocToggle), h(SearchButton)],
            'aside-outline-before': () => h(TocAside),
        })
    },
}
