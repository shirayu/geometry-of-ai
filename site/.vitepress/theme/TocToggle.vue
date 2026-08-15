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
  <button class="toc-toggle" :class="{ active: !hidden }" @click="toggle" :title="hidden ? '目次を表示' : '目次を隠す'" role="switch" :aria-checked="!hidden">
    <span class="toc-toggle-track">
      <span class="toc-toggle-knob">
        <svg class="toc-toggle-icon" xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="3" y1="6" x2="21" y2="6" />
          <line x1="3" y1="12" x2="15" y2="12" />
          <line x1="3" y1="18" x2="18" y2="18" />
        </svg>
      </span>
    </span>
  </button>
</template>

<style scoped>
.toc-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-left: 8px;
  margin-right: 0;
  border: none;
  background: transparent;
  cursor: pointer;
  flex-shrink: 0;
}

.toc-toggle-track {
  position: relative;
  border-radius: 11px;
  display: block;
  width: 40px;
  height: 22px;
  flex-shrink: 0;
  border: 1px solid var(--vp-input-border-color);
  background-color: var(--vp-input-switch-bg-color);
  transition: border-color 0.25s, background-color 0.25s !important;
}

.toc-toggle:hover .toc-toggle-track {
  border-color: var(--vp-c-brand-1);
}

.toc-toggle.active .toc-toggle-track {
  background-color: var(--vp-c-brand-soft);
}

.toc-toggle-knob {
  position: absolute;
  top: 1px;
  left: 1px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background-color: var(--vp-c-neutral-inverse);
  box-shadow: var(--vp-shadow-1);
  transition: transform 0.25s !important;
}

.toc-toggle.active .toc-toggle-knob {
  transform: translateX(18px);
}

.toc-toggle-icon {
  color: var(--vp-c-text-2);
}

.toc-toggle.active .toc-toggle-icon {
  color: var(--vp-c-brand-1);
}
</style>
