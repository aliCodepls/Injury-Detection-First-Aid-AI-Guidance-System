// app/index.tsx — Home / Camera Screen
import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  Image,
  StyleSheet,
  SafeAreaView,
  Alert,
  Dimensions,
  ActivityIndicator,
  Modal,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { analyzeWound, checkHealth } from "../services/api";
import { theme } from "../constants/theme";

const { width } = Dimensions.get("window");
const PREVIEW_H = Math.min(width - theme.padH * 2, 340);

const LOADING_STEPS = [
  "Sending image…",
  "Analyzing wound region…",
  "Classifying injury…",
  "Preparing first aid guidance…",
];

export default function HomeScreen() {
  const router = useRouter();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [serverOnline, setServerOnline] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);

  useEffect(() => {
    checkHealth().then(setServerOnline);
  }, []);

  const pickFromCamera = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission Required", "Camera access is needed to analyze wounds.");
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.85,
      allowsEditing: true,
      aspect: [1, 1],
    });
    if (!result.canceled && result.assets[0]) {
      setSelectedImage(result.assets[0].uri);
    }
  };

  const pickFromGallery = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission Required", "Photo library access is needed.");
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.85,
      allowsEditing: true,
      aspect: [1, 1],
    });
    if (!result.canceled && result.assets[0]) {
      setSelectedImage(result.assets[0].uri);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedImage) return;
    setLoading(true);
    setLoadingStep(0);
    const interval = setInterval(() => {
      setLoadingStep((prev) => Math.min(prev + 1, LOADING_STEPS.length - 1));
    }, 850);

    try {
      const result = await analyzeWound(selectedImage);
      clearInterval(interval);
      setLoading(false);
      router.push({
        pathname: "/result",
        params: { data: JSON.stringify(result), imageUri: selectedImage },
      });
    } catch (err: unknown) {
      clearInterval(interval);
      setLoading(false);
      const message = err instanceof Error ? err.message : "Unknown error";
      Alert.alert(
        "Analysis Failed",
        message + "\n\nEnsure the backend is running and your phone is on the same network.",
        [{ text: "OK" }]
      );
    }
  };

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.logoRow}>
            <View style={styles.logoIcon}>
              <Ionicons name="medkit" size={20} color={theme.primaryDark} />
            </View>
            <Text style={styles.logoText}>
              FirstSight <Text style={styles.logoAccent}>AI</Text>
            </Text>
          </View>

          <View
            style={[
              styles.statusPill,
              {
                backgroundColor:
                  serverOnline === null ? theme.bgMuted : serverOnline ? theme.primarySoft : theme.accentSoft,
                borderColor: serverOnline === false ? "#FECACA" : theme.border,
              },
            ]}
          >
            <View
              style={[
                styles.statusDot,
                {
                  backgroundColor:
                    serverOnline === null ? theme.textMuted : serverOnline ? theme.primary : theme.accent,
                },
              ]}
            />
            <Text style={styles.statusText} numberOfLines={1}>
              {serverOnline === null ? "Checking…" : serverOnline ? "Backend online" : "Backend offline"}
            </Text>
          </View>
        </View>

        {/* Hero — centered to avoid edge clipping */}
        <View style={styles.heroSection}>
          <Text style={styles.heroTitle}>Wound analysis{"\n"}& first aid</Text>
          <Text style={styles.heroSubtitle}>
            Capture or upload a clear photo. Our AI identifies the injury and suggests evidence-based first aid steps.
          </Text>
        </View>

        {/* Image preview */}
        <View style={styles.imageContainer}>
          {selectedImage ? (
            <Image source={{ uri: selectedImage }} style={styles.previewImage} />
          ) : (
            <View style={styles.imagePlaceholder}>
              <Ionicons name="images-outline" size={48} color={theme.textMuted} />
              <Text style={styles.placeholderText}>No image selected</Text>
            </View>
          )}
          {selectedImage && (
            <TouchableOpacity style={styles.clearBtn} onPress={() => setSelectedImage(null)}>
              <Ionicons name="close-circle" size={28} color={theme.accent} />
            </TouchableOpacity>
          )}
        </View>

        {/* Camera / Gallery */}
        <View style={styles.actionRow}>
          <TouchableOpacity style={styles.actionBtn} onPress={pickFromCamera} activeOpacity={0.85}>
            <Ionicons name="camera" size={22} color={theme.primaryDark} />
            <Text style={styles.actionBtnText}>Camera</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.actionBtn} onPress={pickFromGallery} activeOpacity={0.85}>
            <Ionicons name="images" size={22} color={theme.primaryDark} />
            <Text style={styles.actionBtnText}>Gallery</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity
          style={[styles.primaryBtn, !selectedImage && styles.primaryBtnDisabled]}
          onPress={handleAnalyze}
          disabled={!selectedImage || loading}
          activeOpacity={0.9}
        >
          <Text style={styles.primaryBtnText}>Analyze wound</Text>
          <Ionicons name="arrow-forward" size={20} color="#FFF" />
        </TouchableOpacity>

        <Text style={styles.disclaimer}>
          FirstSight AI supports first aid decisions only. For serious or worsening symptoms, seek professional care or emergency services.
        </Text>
      </View>

      <Modal visible={loading} transparent animationType="fade">
        <View style={styles.modalBackdrop}>
          <View style={styles.loadingCard}>
            <ActivityIndicator size="large" color={theme.primary} style={{ marginBottom: 20 }} />
            <Text style={styles.loadingTitle}>Analyzing</Text>
            {LOADING_STEPS.map((step, i) => (
              <View key={step} style={styles.loadingStep}>
                <View style={[styles.stepDot, { backgroundColor: i <= loadingStep ? theme.primary : theme.border }]} />
                <Text
                  style={[styles.stepText, { color: i <= loadingStep ? theme.text : theme.textMuted }]}
                  numberOfLines={2}
                >
                  {step}
                </Text>
              </View>
            ))}
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: theme.bg },
  container: {
    flex: 1,
    paddingHorizontal: theme.padH,
    paddingBottom: 16,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingTop: 12,
    marginBottom: 20,
    gap: 12,
  },
  logoRow: { flexDirection: "row", alignItems: "center", gap: 10, flexShrink: 1 },
  logoIcon: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: theme.primarySoft,
    borderWidth: 1,
    borderColor: theme.border,
    justifyContent: "center",
    alignItems: "center",
  },
  logoText: {
    fontSize: 20,
    fontWeight: "800",
    color: theme.text,
    letterSpacing: -0.3,
  },
  logoAccent: { color: theme.primaryDark },
  statusPill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 999,
    borderWidth: 1,
    maxWidth: width * 0.44,
  },
  statusDot: { width: 8, height: 8, borderRadius: 4 },
  statusText: { fontSize: 12, color: theme.textSecondary, fontWeight: "600", flexShrink: 1 },
  heroSection: {
    marginBottom: 22,
    alignItems: "center",
    paddingHorizontal: 4,
  },
  heroTitle: {
    fontSize: 28,
    fontWeight: "800",
    color: theme.text,
    letterSpacing: -0.8,
    lineHeight: 34,
    marginBottom: 10,
    textAlign: "center",
    alignSelf: "center",
    width: "100%",
  },
  heroSubtitle: {
    fontSize: 15,
    color: theme.textMuted,
    lineHeight: 22,
    textAlign: "center",
    maxWidth: 360,
    alignSelf: "center",
  },
  imageContainer: {
    position: "relative",
    marginBottom: 20,
    alignSelf: "center",
    width: "100%",
    maxWidth: PREVIEW_H + 40,
  },
  previewImage: {
    width: "100%",
    height: PREVIEW_H,
    borderRadius: theme.radiusLg,
    backgroundColor: theme.bgMuted,
  },
  imagePlaceholder: {
    width: "100%",
    height: PREVIEW_H,
    borderRadius: theme.radiusLg,
    backgroundColor: theme.bgElevated,
    borderWidth: 2,
    borderColor: theme.border,
    borderStyle: "dashed",
    justifyContent: "center",
    alignItems: "center",
    gap: 10,
  },
  placeholderText: { color: theme.textMuted, fontSize: 15 },
  clearBtn: { position: "absolute", top: 12, right: 12 },
  actionRow: { flexDirection: "row", gap: 12, marginBottom: 18 },
  actionBtn: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    backgroundColor: theme.bgElevated,
    borderRadius: theme.radiusMd,
    paddingVertical: 15,
    borderWidth: 1,
    borderColor: theme.border,
    shadowColor: theme.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 1,
    shadowRadius: 8,
    elevation: 2,
  },
  actionBtnText: { fontSize: 15, fontWeight: "700", color: theme.text },
  primaryBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    backgroundColor: theme.primary,
    borderRadius: theme.radiusMd,
    paddingVertical: 16,
    marginBottom: 14,
    shadowColor: "#3A5566",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 12,
    elevation: 4,
  },
  primaryBtnDisabled: { backgroundColor: theme.borderStrong, shadowOpacity: 0 },
  primaryBtnText: { fontSize: 16, fontWeight: "800", color: "#FFFFFF" },
  disclaimer: {
    textAlign: "center",
    fontSize: 12,
    color: theme.textMuted,
    lineHeight: 18,
    paddingHorizontal: 8,
  },
  modalBackdrop: {
    flex: 1,
    backgroundColor: "rgba(58, 85, 102, 0.32)",
    justifyContent: "center",
    padding: 24,
  },
  loadingCard: {
    backgroundColor: theme.bgElevated,
    borderRadius: theme.radiusLg,
    padding: 28,
    borderWidth: 1,
    borderColor: theme.border,
    shadowColor: theme.shadow,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 1,
    shadowRadius: 24,
    elevation: 8,
  },
  loadingTitle: {
    fontSize: 20,
    fontWeight: "800",
    color: theme.text,
    marginBottom: 20,
    textAlign: "center",
  },
  loadingStep: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    marginBottom: 12,
  },
  stepDot: { width: 8, height: 8, borderRadius: 4 },
  stepText: { fontSize: 14, flex: 1, flexWrap: "wrap" },
});
