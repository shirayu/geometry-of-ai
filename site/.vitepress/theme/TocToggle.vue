<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useData, useRoute } from 'vitepress'

const { page } = useData()
const route = useRoute()
const hidden = ref(false)
const activeH2 = ref('')
const activeH3 = ref('')

const STORAGE_KEY = 'toc-hidden'

function headingText(el: Element): string {
  return Array.from(el.childNodes)
    .filter(n => !(n instanceof Element && n.tagName === 'A' && n.classList.contains('header-anchor')))
    .map(n => n.textContent ?? '')
    .join('')
    .trim()
}

function updateActive() {
  const active = document.querySelector<HTMLElement>('.VPDocAsideOutline .outline-link.active[href]')
  if (!active) { activeH2.value = ''; activeH3.value = ''; return }
  const id = (active.getAttribute('href') ?? '').slice(1)
  if (!id) { activeH2.value = ''; activeH3.value = ''; return }
  const heading = document.getElementById(id)
  if (!heading) { activeH2.value = ''; activeH3.value = ''; return }

  if (heading.tagName === 'H2') {
    activeH2.value = headingText(heading)
    activeH3.value = ''
  } else if (heading.tagName === 'H3') {
    activeH3.value = headingText(heading)
    let prev = heading.previousElementSibling
    while (prev && prev.tagName !== 'H2') prev = prev.previousElementSibling
    activeH2.value = prev ? headingText(prev) : ''
  } else {
    activeH2.value = ''
    activeH3.value = ''
  }
}

let rafId = 0

function onScroll() {
  cancelAnimationFrame(rafId)
  rafId = requestAnimationFrame(updateActive)
}

onMounted(() => {
  hidden.value = localStorage.getItem(STORAGE_KEY) === '1'
  applyClass(hidden.value)
  window.addEventListener('scroll', onScroll, { passive: true })
  updateActive()
})

onUnmounted(() => {
  window.removeEventListener('scroll', onScroll)
  cancelAnimationFrame(rafId)
})

watch(() => route.path, () => {
  activeH2.value = ''
  activeH3.value = ''
  setTimeout(updateActive, 300)
})

function toggle() {
  hidden.value = !hidden.value
  localStorage.setItem(STORAGE_KEY, hidden.value ? '1' : '0')
  applyClass(hidden.value)
}

function applyClass(hide: boolean) {
  document.documentElement.classList.toggle('hide-toc', hide)
}
</script>

<template>
  <button class="toc-toggle" :class="{ active: hidden }" @click="toggle" :title="hidden ? '目次を表示' : '目次を隠す'">
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <line x1="3" y1="6" x2="21" y2="6" />
      <line x1="3" y1="12" x2="15" y2="12" />
      <line x1="3" y1="18" x2="18" y2="18" />
    </svg>
  </button>
  <span v-if="page.title" class="nav-page-title">
    <span class="nav-h1-title">{{ page.title }}</span>
    <span v-if="activeH2 || activeH3" class="nav-sub-title">
      <span class="nav-h2-title">{{ activeH2 }}</span>
      <span class="nav-h3-title">{{ activeH3 }}</span>
    </span>
  </span>
</template>

<style scoped>
.toc-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-left: 8px;
  margin-right: 0;
  width: 32px;
  height: 32px;
  border-radius: 6px;
  border: none;
  background: transparent;
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: color 0.2s, background-color 0.2s;
  flex-shrink: 0;
}

.toc-toggle:hover {
  color: var(--vp-c-text-1);
  background-color: var(--vp-c-default-soft);
}

.toc-toggle.active {
  color: var(--vp-c-brand-1);
  background-color: var(--vp-c-brand-soft);
}

.nav-page-title {
  margin-left: 12px;
  margin-right: auto;
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 12px;
  overflow: hidden;
}

.nav-h1-title {
  font-size: 13px;
  font-weight: 500;
  color: var(--vp-c-text-2);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex-shrink: 0;
}

.nav-sub-title {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 1px;
  overflow: hidden;
  border-left: 1px solid var(--vp-c-divider);
  padding-left: 12px;
}

.nav-h2-title {
  font-size: 12px;
  font-weight: 500;
  color: var(--vp-c-text-1);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.3;
  min-height: 1.3em;
}

.nav-h3-title {
  font-size: 11px;
  font-weight: 400;
  color: var(--vp-c-text-2);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.3;
  min-height: 1.3em;
}
</style>
