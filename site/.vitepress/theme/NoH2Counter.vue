<script setup lang="ts">
import { watch, onMounted } from 'vue'
import { useRoute } from 'vitepress'

const route = useRoute()

const NO_COUNTER_PATHS = [
  '/',
  '/series/quizzes',
  '/series/keywords',
  '/series/intro',
  '/series/references',
  '/series/toc',
]

function apply() {
  if (typeof document === 'undefined') return
  const path = route.path.replace(/\.html$/, '')
  const noCounter = NO_COUNTER_PATHS.includes(path)
  document.documentElement.classList.toggle('no-h2-counter', noCounter)
}

onMounted(apply)
watch(() => route.path, apply)
</script>

<template></template>
