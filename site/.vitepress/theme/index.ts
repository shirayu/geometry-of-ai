import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import TocToggle from './TocToggle.vue'
import SearchButton from './SearchButton.vue'
import TocAside from './TocAside.vue'
import './custom.css'

export default {
    extends: DefaultTheme,
    Layout: () => {
        return h(DefaultTheme.Layout, null, {
            'nav-bar-content-before': () => [h(TocToggle), h(SearchButton)],
            'aside-outline-before': () => h(TocAside),
        })
    },
}
