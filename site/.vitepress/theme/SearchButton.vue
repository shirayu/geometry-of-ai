<script setup lang="ts">
function openSearch() {
  const btn = document.querySelector<HTMLElement>('.VPNavBarSearch button')
  btn?.click()
}
</script>

<template>
  <button class="search-icon-btn" @click="openSearch" title="検索">
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  </button>
</template>

<style scoped>
.search-icon-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-left: 4px;
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

.search-icon-btn:hover {
  color: var(--vp-c-text-1);
  background-color: var(--vp-c-default-soft);
}
</style>
