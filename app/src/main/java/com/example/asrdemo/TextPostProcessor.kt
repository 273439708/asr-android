package com.example.asrdemo

/**
 * 识别结果文本后处理：
 * 1. 英文大小写规范化（模型输出全大写 → 转小写，句首及独立 "i" 大写）
 * 2. 中英文边界补空格（"你好HELLO世界" → "你好 hello 世界"）
 * 3. 压缩多余空白
 * 4. 标点润色（加标点后调用）：英文后面的全角标点转半角，句中大写修复
 */
object TextPostProcessor {

    // CJK 与 拉丁字母/数字 相邻的边界
    private val cjkThenLatin = Regex("([\\u4e00-\\u9fff])([A-Za-z0-9])")
    private val latinThenCjk = Regex("([A-Za-z0-9])([\\u4e00-\\u9fff])")
    private val multiSpace = Regex("\\s+")
    // 独立的 i（前后是边界）
    private val standaloneI = Regex("\\bi\\b")

    fun process(raw: String): String {
        var text = raw.trim()
        if (text.isEmpty()) return text

        // 模型输出英文为全大写 → 统一小写
        text = text.lowercase()

        // 中英边界加空格
        text = cjkThenLatin.replace(text) { "${it.groupValues[1]} ${it.groupValues[2]}" }
        text = latinThenCjk.replace(text) { "${it.groupValues[1]} ${it.groupValues[2]}" }

        // 压缩空白
        text = multiSpace.replace(text, " ").trim()

        // 独立 i → I
        text = standaloneI.replace(text, "I")

        // 首个拉丁字母大写（句首是中文则跳过）
        val firstLatin = text.indexOfFirst { it in 'a'..'z' || it in 'A'..'Z' }
        if (firstLatin >= 0) {
            // 仅当它之前没有中文字符时才视为"句首英文"
            val hasCjkBefore = text.substring(0, firstLatin).any { it in '一'..'鿿' }
            if (!hasCjkBefore) {
                text = text.substring(0, firstLatin) +
                    text[firstLatin].uppercaseChar() +
                    text.substring(firstLatin + 1)
            }
        }
        return text
    }

    // ---------- 标点润色（在 OfflinePunctuation 之后调用） ----------

    private val fullToHalf = mapOf(
        '。' to '.', '，' to ',', '？' to '?',
        '！' to '!', '：' to ':', '；' to ';',
    )
    // 句末标点 + 空格 + 小写字母 → 新句首大写
    private val sentenceStart = Regex("([.?!]\\s+)([a-z])")

    /**
     * CT-Transformer 词表只有中文全角标点。本函数把"前一个字符是英文/数字"的
     * 全角标点转为半角，并在其后补空格；随后修复英文新句首的大写。
     *
     * "My name is jane。" → "My name is jane."
     * "你好，my name is。" → 保持 "你好，"（前面是中文），结尾 → "my name is."
     */
    fun polishPunctuation(raw: String): String {
        if (raw.isEmpty()) return raw
        val sb = StringBuilder(raw.length + 8)
        for (i in raw.indices) {
            val c = raw[i]
            val half = fullToHalf[c]
            val prev = sb.lastOrNull()
            val prevIsLatin = prev != null &&
                (prev in 'a'..'z' || prev in 'A'..'Z' || prev in '0'..'9')
            if (half != null && prevIsLatin) {
                sb.append(half)
                // 后面紧跟正文时补一个空格
                val next = raw.getOrNull(i + 1)
                if (next != null && next != ' ' && !fullToHalf.containsKey(next)) {
                    sb.append(' ')
                }
            } else {
                sb.append(c)
            }
        }
        var text = sb.toString()
        // 半角句末标点之后的英文新句首大写
        text = sentenceStart.replace(text) {
            it.groupValues[1] + it.groupValues[2].uppercase()
        }
        return text
    }
}
