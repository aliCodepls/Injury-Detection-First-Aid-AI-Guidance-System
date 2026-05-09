// app/result.tsx — Results Screen
import React, { useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  Image,
  StyleSheet,
  SafeAreaView,
  ScrollView,
} from "react-native";
import { useRouter, useLocalSearchParams } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { AnalyzeResponse } from "../services/api";
import { theme } from "../constants/theme";
import { getWoundIconVisual } from "../constants/woundIcons";

const SEVERITY_COLORS: Record<string, string> = {
  mild: theme.success,
  moderate: theme.warning,
  severe: theme.accent,
};

export default function ResultScreen() {
  const router = useRouter();
  const { data, imageUri } = useLocalSearchParams<{ data: string; imageUri: string }>();
  const [activeTab, setActiveTab] = useState<"steps" | "donot">("steps");

  if (!data) {
    return (
      <SafeAreaView style={styles.safe}>
        <Text style={styles.emptyText}>No result data found.</Text>
      </SafeAreaView>
    );
  }

  const result: AnalyzeResponse = JSON.parse(data);
  const severityColor = SEVERITY_COLORS[result.severity_level] ?? theme.warning;
  const woundIcon = getWoundIconVisual(result.wound_type);
  const confidencePct = Math.round(result.confidence * 100);

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {/* Top bar */}
        <View style={styles.topBar}>
          <TouchableOpacity style={styles.iconBtn} onPress={() => router.push("/")} activeOpacity={0.85}>
            <Ionicons name="arrow-back" size={22} color={theme.text} />
          </TouchableOpacity>
          <Text style={styles.screenTitle}>Analysis</Text>
          <View style={{ width: 44 }} />
        </View>

        {result.seek_emergency && (
          <View style={styles.emergencyBanner}>
            <Ionicons name="warning" size={22} color={theme.accent} />
            <Text style={styles.emergencyText}>
              This injury may need urgent care. Contact emergency services if symptoms are severe or worsening.
            </Text>
          </View>
        )}

        <View style={styles.resultCard}>
          {imageUri ? <Image source={{ uri: imageUri }} style={styles.resultImage} /> : null}

          <View style={styles.cardBody}>
            <View style={styles.woundHeaderOuter}>
              <View style={styles.woundHeader}>
                <View style={[styles.iconCircle, { backgroundColor: woundIcon.circleBg }]}>
                  <Ionicons name={woundIcon.name} size={28} color={woundIcon.iconColor} />
                </View>
                <View style={styles.woundTitleBlock}>
                  <Text style={styles.woundLabel} numberOfLines={3}>
                    {result.wound_label}
                  </Text>
                  <View style={styles.tagRow}>
                    <View
                      style={[
                        styles.tag,
                        {
                          borderColor: severityColor + "55",
                          backgroundColor: severityColor + "18",
                        },
                      ]}
                    >
                      <Text style={[styles.tagText, { color: severityColor }]}>
                        {result.severity_level.toUpperCase()}
                      </Text>
                    </View>
                  </View>
                </View>
              </View>
            </View>

            <View style={styles.confidenceSection}>
              <View style={styles.confidenceHeader}>
                <Text style={styles.confidenceLabel}>Model confidence</Text>
                <Text style={styles.confidenceValue}>{result.confidence_percent}</Text>
              </View>
              <View style={styles.confidenceTrack}>
                <View style={[styles.confidenceFill, { width: `${confidencePct}%` as `${number}%` }]} />
              </View>
            </View>
          </View>
        </View>

        <View style={styles.tabRow}>
          <TouchableOpacity
            style={[styles.tab, activeTab === "steps" && styles.tabActivePrimary]}
            onPress={() => setActiveTab("steps")}
            activeOpacity={0.9}
          >
            <Ionicons
              name="list"
              size={18}
              color={activeTab === "steps" ? theme.primaryDark : theme.textMuted}
            />
            <Text style={[styles.tabText, activeTab === "steps" && styles.tabTextPrimary]}>First aid steps</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.tab, activeTab === "donot" && styles.tabActiveWarn]}
            onPress={() => setActiveTab("donot")}
            activeOpacity={0.9}
          >
            <Ionicons
              name="hand-left"
              size={18}
              color={activeTab === "donot" ? theme.accent : theme.textMuted}
            />
            <Text style={[styles.tabText, activeTab === "donot" && styles.tabTextWarn]}>Avoid</Text>
          </TouchableOpacity>
        </View>

        {activeTab === "steps" && (
          <View style={styles.stepsContainer}>
            {result.first_aid.steps.map((step, i) => (
              <View key={i} style={styles.stepCard}>
                <View style={styles.stepNumber}>
                  <Text style={styles.stepNumberText}>{i + 1}</Text>
                </View>
                <Text style={styles.stepBody}>{step}</Text>
              </View>
            ))}
          </View>
        )}

        {activeTab === "donot" && (
          <View style={styles.stepsContainer}>
            {result.first_aid.do_not.map((item, i) => (
              <View key={i} style={[styles.stepCard, styles.doNotCard]}>
                <Ionicons name="close-circle" size={22} color={theme.accent} style={{ marginTop: 2 }} />
                <Text style={styles.stepBody}>{item}</Text>
              </View>
            ))}
          </View>
        )}

        <TouchableOpacity style={styles.secondaryBtn} onPress={() => router.push("/")} activeOpacity={0.9}>
          <Ionicons name="refresh" size={20} color={theme.primaryDark} />
          <Text style={styles.secondaryBtnText}>New analysis</Text>
        </TouchableOpacity>

        <Text style={styles.disclaimer}>
          Educational first aid guidance only. Always follow local protocols and consult a clinician when unsure.
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: theme.bg },
  scrollContent: {
    paddingHorizontal: theme.padH,
    paddingBottom: 32,
    paddingTop: 8,
  },
  emptyText: {
    color: theme.textMuted,
    textAlign: "center",
    marginTop: 48,
    fontSize: 16,
  },
  topBar: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 16,
  },
  iconBtn: {
    width: 44,
    height: 44,
    borderRadius: 12,
    backgroundColor: theme.bgElevated,
    borderWidth: 1,
    borderColor: theme.border,
    justifyContent: "center",
    alignItems: "center",
  },
  screenTitle: {
    fontSize: 18,
    fontWeight: "800",
    color: theme.text,
    textAlign: "center",
    flex: 1,
  },
  emergencyBanner: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 12,
    backgroundColor: theme.accentSoft,
    borderRadius: theme.radiusMd,
    padding: 14,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#FECACA",
  },
  emergencyText: {
    flex: 1,
    color: "#991B1B",
    fontSize: 14,
    lineHeight: 21,
    textAlign: "left",
  },
  resultCard: {
    backgroundColor: theme.bgElevated,
    borderRadius: theme.radiusLg,
    overflow: "hidden",
    marginBottom: 18,
    borderWidth: 1,
    borderColor: theme.border,
    shadowColor: theme.shadow,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 1,
    shadowRadius: 16,
    elevation: 3,
  },
  resultImage: {
    width: "100%",
    height: 220,
    backgroundColor: theme.bgMuted,
  },
  cardBody: {
    paddingVertical: 20,
    paddingHorizontal: 20,
  },
  woundHeaderOuter: {
    width: "100%",
    alignItems: "center",
  },
  woundHeader: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 16,
    width: "100%",
    maxWidth: 400,
    alignSelf: "center",
    paddingHorizontal: 2,
  },
  iconCircle: {
    width: 56,
    height: 56,
    borderRadius: 18,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 1,
    borderColor: theme.border,
  },
  woundTitleBlock: {
    flex: 1,
    minWidth: 0,
    alignItems: "stretch",
    justifyContent: "center",
    paddingTop: 2,
  },
  woundLabel: {
    fontSize: 22,
    fontWeight: "800",
    color: theme.text,
    marginBottom: 8,
    textAlign: "left",
    flexWrap: "wrap",
    width: "100%",
  },
  tagRow: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  tag: {
    alignSelf: "flex-start",
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 999,
    borderWidth: 1,
  },
  tagText: { fontSize: 11, fontWeight: "800", letterSpacing: 0.6 },
  confidenceSection: {
    marginTop: 18,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: theme.border,
  },
  confidenceHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 8,
  },
  confidenceLabel: {
    fontSize: 13,
    color: theme.textMuted,
    fontWeight: "600",
  },
  confidenceValue: {
    fontSize: 13,
    color: theme.text,
    fontWeight: "800",
  },
  confidenceTrack: {
    height: 8,
    backgroundColor: theme.bgMuted,
    borderRadius: 4,
    overflow: "hidden",
  },
  confidenceFill: {
    height: 8,
    backgroundColor: theme.primary,
    borderRadius: 4,
  },
  tabRow: {
    flexDirection: "row",
    gap: 10,
    marginBottom: 16,
  },
  tab: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    paddingVertical: 14,
    borderRadius: theme.radiusMd,
    backgroundColor: theme.bgElevated,
    borderWidth: 1,
    borderColor: theme.border,
  },
  tabActivePrimary: {
    backgroundColor: theme.primarySoft,
    borderColor: theme.primary + "66",
  },
  tabActiveWarn: {
    backgroundColor: theme.accentSoft,
    borderColor: "#FECACA",
  },
  tabText: {
    fontSize: 14,
    fontWeight: "700",
    color: theme.textMuted,
  },
  tabTextPrimary: { color: theme.primaryDark },
  tabTextWarn: { color: theme.accent },
  stepsContainer: { gap: 10, marginBottom: 20 },
  stepCard: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 14,
    backgroundColor: theme.bgElevated,
    borderRadius: theme.radiusMd,
    padding: 16,
    borderWidth: 1,
    borderColor: theme.border,
  },
  doNotCard: {
    borderColor: "#FECACA",
    backgroundColor: "#FFFBFB",
  },
  stepNumber: {
    width: 28,
    height: 28,
    borderRadius: 8,
    backgroundColor: theme.primarySoft,
    justifyContent: "center",
    alignItems: "center",
  },
  stepNumberText: {
    fontSize: 13,
    fontWeight: "900",
    color: theme.primaryDark,
  },
  stepBody: {
    flex: 1,
    fontSize: 15,
    color: theme.textSecondary,
    lineHeight: 24,
    textAlign: "left",
  },
  secondaryBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    backgroundColor: theme.bgElevated,
    borderRadius: theme.radiusMd,
    paddingVertical: 15,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: theme.border,
  },
  secondaryBtnText: {
    fontSize: 15,
    fontWeight: "700",
    color: theme.primaryDark,
  },
  disclaimer: {
    textAlign: "center",
    fontSize: 12,
    color: theme.textMuted,
    lineHeight: 18,
    paddingHorizontal: 8,
  },
});
