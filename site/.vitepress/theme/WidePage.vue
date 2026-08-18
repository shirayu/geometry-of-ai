<script setup lang="ts">
import { watch, onMounted } from 'vue'
import { useRoute } from 'vitepress'

const route = useRoute()

const WIDE_PATHS = ['/']

function apply() {
  if (typeof document === 'undefined') return
  const path = route.path.replace(/\.html$/, '')
  const wide = WIDE_PATHS.includes(path)
  document.documentElement.classList.toggle('wide-page', wide)
}

onMounted(apply)
watch(() => route.path, apply)
</script>

<template>
  <!-- VitePress 1で空コンポーネントがSSRのコメントノードをずらすための安定要素。VitePress 2移行時も要再検証。 -->
  <span class="layout-effect" aria-hidden="true" />
</template>
