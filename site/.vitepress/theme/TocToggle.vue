<script setup lang="ts">
import { ref, onMounted } from 'vue'

const hidden = ref(false)

const STORAGE_KEY = 'toc-hidden'

onMounted(() => {
  hidden.value = localStorage.getItem(STORAGE_KEY) === '1'
  applyClass(hidden.value)
})

function toggle() {
  if (window.matchMedia('(max-width: 959px)').matches) {
    document.querySelector<HTMLElement>('.VPLocalNav .menu')?.click()
    return
  }
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
</style>
