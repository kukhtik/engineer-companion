package com.varian.engcomp.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.varian.engcomp.model.SearchResult
import com.varian.engcomp.ui.theme.DribbbleDark

@Composable
fun SourcesList(
    sources: List<SearchResult>,
    modifier: Modifier = Modifier
) {
    if (sources.isEmpty()) return

    Column(modifier = modifier.padding(horizontal = 16.dp)) {
        Text(
            text = "Источники (${sources.size}):",
            color = DribbbleDark.textMuted,
            fontSize = 12.sp,
            modifier = Modifier.padding(bottom = 4.dp)
        )

        sources.forEach { src ->
            Surface(
                shape = RoundedCornerShape(6.dp),
                color = DribbbleDark.bgSurface,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 2.dp)
            ) {
                Column(modifier = Modifier.padding(8.dp)) {
                    Text(
                        text = src.source,
                        color = DribbbleDark.textSecondary,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium
                    )
                    Text(
                        text = "стр.${src.page} — ${src.section}",
                        color = DribbbleDark.textMuted,
                        fontSize = 10.sp
                    )
                }
            }
        }
    }
}
