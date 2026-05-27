<script setup lang="ts">
import { onMounted, ref } from 'vue'

const props = defineProps<{
    codeBase64: string
}>()

const container = ref<HTMLElement | null>(null)
const error = ref('')

function decodeBase64Utf8(value: string) {
    const binary = atob(value)
    const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0))
    return new TextDecoder().decode(bytes)
}

onMounted(async () => {
    if (!container.value) return

    try {
        const mermaid = (await import('mermaid')).default
        mermaid.initialize({
            startOnLoad: false,
            securityLevel: 'loose',
        })

        const id = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2)}`
        const { svg } = await mermaid.render(id, decodeBase64Utf8(props.codeBase64))
        container.value.innerHTML = svg
    } catch (err) {
        error.value = err instanceof Error ? err.message : String(err)
    }
})
</script>

<template>
    <figure class="mermaid-diagram">
        <div ref="container" class="mermaid-diagram__canvas" />
        <pre v-if="error" class="mermaid-diagram__error">{{ error }}</pre>
    </figure>
</template>
