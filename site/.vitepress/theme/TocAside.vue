<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useRoute } from 'vitepress'
import { data } from '../../series/toc.data.ts'

const route = useRoute()
const isTocPage = () => route.path === '/series/toc' || route.path === '/series/toc.html'

const activeId = ref('')

// 部ごとに循環する色相（寒色→暖色）。Appendixもグラデーションの最後に含める。
const PART_HUES = [210, 260, 310, 350, 20, 40, 90]

function partHue(part?: string): number | null {
  if (!part) return null
  const order = [...new Set(data.map(p => p.part).filter((p): p is string => !!p))]
  const i = order.indexOf(part)
  return PART_HUES[i % PART_HUES.length]
}

let observer: IntersectionObserver | null = null

function setupObserver() {
    if (!isTocPage()) return
    observer?.disconnect()

    const ids = data.map(p => p.id)
    const entries = new Map<string, number>()

    observer = new IntersectionObserver(
        (records) => {
            for (const r of records) {
                entries.set(r.target.id, r.intersectionRatio)
            }
            let best = ''
            let bestRatio = -1
            for (const id of ids) {
                const ratio = entries.get(id) ?? 0
                if (ratio > bestRatio) { bestRatio = ratio; best = id }
            }
            if (best) activeId.value = best
        },
        { threshold: [0, 0.1, 0.5, 1.0] }
    )

    for (const id of ids) {
        const el = document.getElementById(id)
        if (el) observer.observe(el)
    }
}

onMounted(setupObserver)
onUnmounted(() => observer?.disconnect())
watch(() => route.path, () => {
    activeId.value = ''
    setTimeout(setupObserver, 300)
})
</script>

<template>
  <div v-if="isTocPage()" class="toc-aside">
    <div class="toc-aside-title">ページ一覧</div>
    <nav>
      <template v-for="(page, index) in data" :key="page.id">
        <div
          v-if="page.part && (index === 0 || data[index - 1].part !== page.part)"
          class="toc-aside-part"
          :style="partHue(page.part) !== null ? { '--part-hue': partHue(page.part) } : {}"
        >{{ page.part }}</div>
        <a
          :href="'#' + page.id"
          class="toc-aside-link"
          :class="{ active: activeId === page.id }"
        >{{ page.title }}</a>
      </template>
    </nav>
  </div>
</template>

<style scoped>
.toc-aside {
  padding: 0 0 1rem;
}

.toc-aside-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--vp-c-text-2);
  margin-bottom: 0.5rem;
  padding: 0 0 0 12px;
}

.toc-aside-part {
  font-size: 12px;
  font-weight: 600;
  color: var(--vp-c-text-3);
  margin-top: 10px;
  padding: 0 0 2px 12px;
}

.toc-aside-part:first-child {
  margin-top: 0;
}

.toc-aside-part[style*="--part-hue"] {
  color: hsl(var(--part-hue), 45%, 42%);
}

.dark .toc-aside-part[style*="--part-hue"] {
  color: hsl(var(--part-hue), 55%, 72%);
}

.toc-aside-link {
  display: block;
  padding: 2px 0 2px 20px;
  font-size: 13px;
  color: var(--vp-c-text-2);
  text-decoration: none;
  line-height: 1.6;
  border-radius: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  transition: color 0.25s;
}

.toc-aside-link:hover {
  color: var(--vp-c-text-1);
}

.toc-aside-link.active {
  background-color: #fef9c3;
  border-radius: 4px;
  color: var(--vp-c-text-1);
}

.dark .toc-aside-link.active {
  background-color: #3d3200;
}
</style>
