import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:http_parser/http_parser.dart';
import 'package:path/path.dart' as p;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fitness ML',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2F8F83),
          brightness: Brightness.light,
        ),
        scaffoldBackgroundColor: const Color(0xFFF7F5FC),
        appBarTheme: const AppBarTheme(
          elevation: 0,
          scrolledUnderElevation: 0,
          backgroundColor: Colors.transparent,
          foregroundColor: Colors.white,
          surfaceTintColor: Colors.transparent,
          toolbarHeight: 82,
        ),
        cardTheme: CardThemeData(
          elevation: 0,
          color: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
          ),
          margin: EdgeInsets.zero,
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            minimumSize: const Size.fromHeight(52),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(18),
            ),
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),
      home: const FirstScreen(),
    );
  }
}

class CVQuality {
  final bool bodyDetected;
  final bool bodyFullyVisible;
  final double photoQualityScore;
  final double keypointConfidenceMean;
  final double visibleKeypointsRatio;

  CVQuality({
    required this.bodyDetected,
    required this.bodyFullyVisible,
    required this.photoQualityScore,
    required this.keypointConfidenceMean,
    required this.visibleKeypointsRatio,
  });

  factory CVQuality.fromJson(Map<String, dynamic> json) {
    return CVQuality(
      bodyDetected: json['body_detected'] == true,
      bodyFullyVisible: json['body_fully_visible'] == true,
      photoQualityScore: (json['photo_quality_score'] as num?)?.toDouble() ?? 0.0,
      keypointConfidenceMean: (json['keypoint_confidence_mean'] as num?)?.toDouble() ?? 0.0,
      visibleKeypointsRatio: (json['visible_keypoints_ratio'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class CVKeypoint {
  final double x;
  final double y;
  final double? z;
  final double confidence;

  CVKeypoint({
    required this.x,
    required this.y,
    required this.z,
    required this.confidence,
  });

  factory CVKeypoint.fromJson(Map<String, dynamic> json) {
    return CVKeypoint(
      x: (json['x'] as num?)?.toDouble() ?? 0.0,
      y: (json['y'] as num?)?.toDouble() ?? 0.0,
      z: (json['z'] as num?)?.toDouble(),
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class CVFeatures {
  final double torsoTiltDeg;
  final double shoulderTiltDeg;
  final double pelvisTiltDeg;
  final double leftKneeAngleDeg;
  final double rightKneeAngleDeg;
  final double leftHipAngleDeg;
  final double rightHipAngleDeg;
  final double shoulderWidthRatio;
  final double hipWidthRatio;
  final double torsoLengthRatio;
  final double legLengthRatio;
  final double shoulderAsymmetry;
  final double pelvisAsymmetry;

  CVFeatures({
    required this.torsoTiltDeg,
    required this.shoulderTiltDeg,
    required this.pelvisTiltDeg,
    required this.leftKneeAngleDeg,
    required this.rightKneeAngleDeg,
    required this.leftHipAngleDeg,
    required this.rightHipAngleDeg,
    required this.shoulderWidthRatio,
    required this.hipWidthRatio,
    required this.torsoLengthRatio,
    required this.legLengthRatio,
    required this.shoulderAsymmetry,
    required this.pelvisAsymmetry,
  });

  factory CVFeatures.fromJson(Map<String, dynamic> json) {
    double getNum(String key) => (json[key] as num?)?.toDouble() ?? 0.0;

    return CVFeatures(
      torsoTiltDeg: getNum('torso_tilt_deg'),
      shoulderTiltDeg: getNum('shoulder_tilt_deg'),
      pelvisTiltDeg: getNum('pelvis_tilt_deg'),
      leftKneeAngleDeg: getNum('left_knee_angle_deg'),
      rightKneeAngleDeg: getNum('right_knee_angle_deg'),
      leftHipAngleDeg: getNum('left_hip_angle_deg'),
      rightHipAngleDeg: getNum('right_hip_angle_deg'),
      shoulderWidthRatio: getNum('shoulder_width_ratio'),
      hipWidthRatio: getNum('hip_width_ratio'),
      torsoLengthRatio: getNum('torso_length_ratio'),
      legLengthRatio: getNum('leg_length_ratio'),
      shoulderAsymmetry: getNum('shoulder_asymmetry'),
      pelvisAsymmetry: getNum('pelvis_asymmetry'),
    );
  }
}

class CVAnalysisResult {
  final String status;
  final CVQuality quality;
  final Map<String, CVKeypoint> keypoints;
  final CVFeatures features;
  final List<String> warnings;

  CVAnalysisResult({
    required this.status,
    required this.quality,
    required this.keypoints,
    required this.features,
    required this.warnings,
  });

  factory CVAnalysisResult.fromJson(Map<String, dynamic> json) {
    final keypointsJson = (json['keypoints'] as Map<String, dynamic>? ?? {});
    final keypoints = keypointsJson.map(
      (key, value) => MapEntry(
        key,
        CVKeypoint.fromJson(value as Map<String, dynamic>),
      ),
    );

    return CVAnalysisResult(
      status: (json['status'] ?? '').toString(),
      quality: CVQuality.fromJson(json['quality'] as Map<String, dynamic>? ?? {}),
      keypoints: keypoints,
      features: CVFeatures.fromJson(json['features'] as Map<String, dynamic>? ?? {}),
      warnings: (json['warnings'] as List<dynamic>? ?? []).map((e) => e.toString()).toList(),
    );
  }
}



class SurrogateInputData {
  final double lx;
  final double ly;
  final double lz;
  final double e;
  final double nu;
  final double tx;
  final double ty;
  final double tz;

  SurrogateInputData({
    required this.lx,
    required this.ly,
    required this.lz,
    required this.e,
    required this.nu,
    required this.tx,
    required this.ty,
    required this.tz,
  });

  factory SurrogateInputData.fromJson(Map<String, dynamic> json) {
    double getNum(String key) => (json[key] as num?)?.toDouble() ?? 0.0;

    return SurrogateInputData(
      lx: getNum('Lx'),
      ly: getNum('Ly'),
      lz: getNum('Lz'),
      e: getNum('E'),
      nu: getNum('nu'),
      tx: getNum('tx'),
      ty: getNum('ty'),
      tz: getNum('tz'),
    );
  }
}

class SurrogatePredictionData {
  final double umax;
  final double u;
  final double sigmavmMax;
  final double rx;

  SurrogatePredictionData({
    required this.umax,
    required this.u,
    required this.sigmavmMax,
    required this.rx,
  });

  factory SurrogatePredictionData.fromJson(Map<String, dynamic> json) {
    double getNum(String key) => (json[key] as num?)?.toDouble() ?? 0.0;

    return SurrogatePredictionData(
      umax: getNum('umax'),
      u: getNum('U'),
      sigmavmMax: getNum('sigmavm_max'),
      rx: getNum('Rx'),
    );
  }
}

class SurrogateInterpretationData {
  final int loadScore;
  final String level;
  final String summary;
  final String progressionHint;

  SurrogateInterpretationData({
    required this.loadScore,
    required this.level,
    required this.summary,
    required this.progressionHint,
  });

  factory SurrogateInterpretationData.fromJson(Map<String, dynamic> json) {
    return SurrogateInterpretationData(
      loadScore: (json['load_score'] as num?)?.toInt() ?? 0,
      level: (json['level'] ?? '').toString(),
      summary: (json['summary'] ?? '').toString(),
      progressionHint: (json['progression_hint'] ?? '').toString(),
    );
  }
}

class SurrogateResult {
  final String status;
  final SurrogateInputData surrogateInput;
  final SurrogatePredictionData prediction;
  final SurrogateInterpretationData interpretation;
  final List<String> warnings;
  final Map<String, dynamic> metadata;

  SurrogateResult({
    required this.status,
    required this.surrogateInput,
    required this.prediction,
    required this.interpretation,
    required this.warnings,
    required this.metadata,
  });

  factory SurrogateResult.fromJson(Map<String, dynamic> json) {
    return SurrogateResult(
      status: (json['status'] ?? '').toString(),
      surrogateInput: SurrogateInputData.fromJson(
        json['surrogate_input'] as Map<String, dynamic>? ?? {},
      ),
      prediction: SurrogatePredictionData.fromJson(
        json['prediction'] as Map<String, dynamic>? ?? {},
      ),
      interpretation: SurrogateInterpretationData.fromJson(
        json['interpretation'] as Map<String, dynamic>? ?? {},
      ),
      warnings: (json['warnings'] as List<dynamic>? ?? [])
          .map((e) => e.toString())
          .toList(),
      metadata: Map<String, dynamic>.from(
        json['metadata'] as Map<String, dynamic>? ?? {},
      ),
    );
  }
}
class FirstScreen extends StatefulWidget {
  const FirstScreen({super.key});

  @override
  State<FirstScreen> createState() => _FirstScreenState();
}

class _FirstScreenState extends State<FirstScreen> {
  static const String _apiBaseUrl = 'http://127.0.0.1:8000';
  static const String _predictPath = '/api/v1/cv/analyze';
  static const String _surrogatePath = '/api/v1/surrogate/predict';

  final _formKey = GlobalKey<FormState>();
  final _ageController = TextEditingController();
  final _heightController = TextEditingController();
  final _weightController = TextEditingController();

  String _gender = 'female'; // ожидается 'female' или 'male'
  String _goal = 'muscle_gain';
  String _experienceLevel = 'beginner';
  File? _imageFile;

  bool _loading = false;
  String? _error;
  CVAnalysisResult? _result;
  SurrogateResult? _surrogateResult;
  String? _surrogateError;

  final ImagePicker _picker = ImagePicker();

  @override
  void dispose() {
    _ageController.dispose();
    _heightController.dispose();
    _weightController.dispose();
    super.dispose();
  }

  Future<void> _pickImage() async {
    setState(() {
      _error = null;
      _result = null;
      _surrogateResult = null;
      _surrogateError = null;
    });

    final XFile? xfile = await _picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 90,
      maxWidth: 1200,
    );

    if (xfile == null) return;

    setState(() {
      _imageFile = File(xfile.path);
    });
  }
  void _removeImage() {
    setState(() {
      _imageFile = null;

      // Чтобы результат не “оставался” от предыдущего фото
      _result = null;
      _surrogateResult = null;
      _surrogateError = null;
      _error = null;
    });
  }
  Future<CVAnalysisResult> _predictCv({
    required File image,
    required String gender,
    required int age,
  }) async {
    final uri = Uri.parse('$_apiBaseUrl$_predictPath');

    final ext = p.extension(image.path).toLowerCase();
    final mediaType = ext == '.png'
        ? MediaType('image', 'png')
        : MediaType('image', 'jpeg');
        
    final request = http.MultipartRequest('POST', uri)
      ..fields['gender'] = gender
      ..fields['age'] = age.toString()
      ..files.add(await http.MultipartFile.fromPath('image', image.path, contentType: mediaType),);

    final streamed = await request.send();
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode != 200) {
      throw Exception(_extractApiError(response));
    }

    final Map<String, dynamic> jsonBody =
        jsonDecode(response.body) as Map<String, dynamic>;


    return CVAnalysisResult.fromJson(jsonBody);
  }

  Future<SurrogateResult> _predictSurrogate({
    required CVFeatures features,
    required String gender,
    required int age,
    required double heightCm,
    required double weightKg,
    required String goal,
    required String experienceLevel,
  }) async {
    final uri = Uri.parse('$_apiBaseUrl$_surrogatePath');

    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'questionnaire': {
          'gender': gender,
          'age': age,
          'height_cm': heightCm,
          'weight_kg': weightKg,
          'goal': goal,
          'experience_level': experienceLevel,
        },
        'features': {
          'torso_tilt_deg': features.torsoTiltDeg,
          'shoulder_tilt_deg': features.shoulderTiltDeg,
          'pelvis_tilt_deg': features.pelvisTiltDeg,
          'left_knee_angle_deg': features.leftKneeAngleDeg,
          'right_knee_angle_deg': features.rightKneeAngleDeg,
          'left_hip_angle_deg': features.leftHipAngleDeg,
          'right_hip_angle_deg': features.rightHipAngleDeg,
          'shoulder_width_ratio': features.shoulderWidthRatio,
          'hip_width_ratio': features.hipWidthRatio,
          'torso_length_ratio': features.torsoLengthRatio,
          'leg_length_ratio': features.legLengthRatio,
          'shoulder_asymmetry': features.shoulderAsymmetry,
          'pelvis_asymmetry': features.pelvisAsymmetry,
        },
      }),
    );

    if (response.statusCode != 200) {
      throw Exception(_extractApiError(response));
    }

    final Map<String, dynamic> jsonBody =
        jsonDecode(response.body) as Map<String, dynamic>;

    return SurrogateResult.fromJson(jsonBody);
  }

  Future<void> _onSubmit() async {
    setState(() {
      _error = null;
      _result = null;
      _surrogateResult = null;
      _surrogateError = null;
    });

    if (!_formKey.currentState!.validate()) return;
    if (_imageFile == null) {
      setState(() => _error = 'Не выбрано изображение.');
      return;
    }

    final int age = int.parse(_ageController.text.trim());
    final double heightCm = double.parse(_heightController.text.trim().replaceAll(',', '.'));
    final double weightKg = double.parse(_weightController.text.trim().replaceAll(',', '.'));

    setState(() => _loading = true);
    try {
      final cvRes = await _predictCv(image: _imageFile!, gender: _gender, age: age);
      setState(() => _result = cvRes);

      try {
        final surrogateRes = await _predictSurrogate(
          features: cvRes.features,
          gender: _gender,
          age: age,
          heightCm: heightCm,
          weightKg: weightKg,
          goal: _goal,
          experienceLevel: _experienceLevel,
        );
        setState(() => _surrogateResult = surrogateRes);
      } catch (e) {
        setState(() => _surrogateError = _humanizeError(e.toString()));
      }
    } catch (e) {
      setState(() => _error = _humanizeError(e.toString()));
    } finally {
      setState(() => _loading = false);
    }
  }

  void _goNext() {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => PlanScreenStub(
          result: _result!,
          surrogateResult: _surrogateResult,
        ),
      ),
    );
  }
  void _showBmiInfo() {
    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Оценка BMI и персонализация'),
          content: const SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Приложение использует пол, возраст и фото для оценки категории BMI '
                  'и последующей персонализации рекомендаций.',
                ),
                SizedBox(height: 12),
                Text('Рекомендации к фото:'),
                SizedBox(height: 6),
                Text('• Хорошее освещение, без сильных теней.'),
                Text('• Минимум посторонних объектов на фоне.'),
                Text('• Желательно, чтобы фигура занимала заметную часть кадра.'),
                SizedBox(height: 12),
                Text(
                  'Результат предназначен для предварительной оценки и может иметь погрешность.',
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Закрыть'),
            ),
          ],
        );
      },
    );
  }
  // Функция для отображения руководства
  void _showUserGuide() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Справка по фотоанализу'),
          content: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                Text(
                  'Требования к фотографии',
                  style: TextStyle(fontWeight: FontWeight.w700),
                ),
                SizedBox(height: 8),
                Text('• Фото должно быть в полный рост.'),
                Text('• Поза — стоя, в естественном положении, без сильного наклона.'),
                Text('• Камера должна быть расположена напротив человека.'),
                Text('• Желательно, чтобы в кадре были полностью видны плечи, таз, колени и стопы.'),
                Text('• Освещение должно быть ровным, без сильных теней и пересветов.'),
                Text('• Одежда не должна полностью скрывать контуры тела.'),
                Text('• В кадре не должно быть крупных посторонних объектов, перекрывающих тело.'),

                SizedBox(height: 16),
                Text(
                  'Что означают показатели анализа',
                  style: TextStyle(fontWeight: FontWeight.w700),
                ),
                SizedBox(height: 8),
                Text(
                  'Качество фото — интегральная оценка пригодности снимка для анализа позы.',
                ),
                SizedBox(height: 6),
                Text(
                  'Средняя уверенность keypoints — средняя уверенность модели в корректности найденных ключевых точек тела '
                  '(например, плеч, таза, коленей). Чем выше значение, тем надежнее распознавание.',
                ),
                SizedBox(height: 6),
                Text(
                  'Тело найдено — модель обнаружила человека на изображении.',
                ),
                SizedBox(height: 6),
                Text(
                  'Тело полностью в кадре — основные анатомические ориентиры присутствуют в поле зрения камеры.',
                ),
                SizedBox(height: 6),
                Text(
                  'Наклон корпуса, плеч и таза — геометрические признаки, отражающие возможную асимметрию положения тела.',
                ),
                SizedBox(height: 6),
                Text(
                  'Углы коленей — расчетные суставные углы, используемые для оценки положения нижних конечностей.',
                ),

                SizedBox(height: 16),
                Text(
                  'Ограничения',
                  style: TextStyle(fontWeight: FontWeight.w700),
                ),
                SizedBox(height: 8),
                Text(
                  'Результат предназначен для предварительной цифровой оценки и не заменяет консультацию врача или тренера.',
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Закрыть'),
            ),
          ],
        );
      },
    );
  }
  String _yesNo(bool value) => value ? 'Да' : 'Нет';

  String _asPercent(double value) => '${(value * 100).toStringAsFixed(0)}%';

  String _qualityLabel(double score) {
    if (score >= 0.85) return 'Высокое';
    if (score >= 0.70) return 'Хорошее';
    if (score >= 0.55) return 'Приемлемое';
    return 'Низкое';
  }

  String _goalLabel(String goal) {
    switch (goal) {
      case 'fat_loss':
        return 'Снижение жировой массы';
      case 'maintenance':
        return 'Поддержание формы';
      case 'recomposition':
        return 'Рекомпозиция';
      case 'muscle_gain':
      default:
        return 'Набор мышечной массы';
    }
  }

  String _experienceLabel(String level) {
    switch (level) {
      case 'intermediate':
        return 'Средний';
      case 'advanced':
        return 'Продвинутый';
      case 'beginner':
      default:
        return 'Начальный';
    }
  }

  String _surrogateLevelLabel(String level) {
    switch (level) {
      case 'low':
        return 'Низкий';
      case 'moderate':
        return 'Умеренный';
      case 'high':
        return 'Высокий';
      default:
        return level;
    }
  }

  Color _loadScoreColor(BuildContext context, int score) {
    if (score < 35) return Colors.green.shade700;
    if (score < 70) return Colors.orange.shade700;
    return Theme.of(context).colorScheme.error;
  }

  String _formatCompact(double value) {
    if (value.abs() >= 1000) return value.toStringAsFixed(0);
    if (value.abs() >= 10) return value.toStringAsFixed(2);
    return value.toStringAsFixed(4);
  }

  String _extractApiError(http.Response response) {
    try {
      final body = jsonDecode(response.body);

      if (body is Map<String, dynamic>) {
        final detail = body['detail']?.toString().trim();
        if (detail != null && detail.isNotEmpty) {
          return _humanizeError(detail);
        }

        final error = body['error']?.toString().trim();
        if (error != null && error.isNotEmpty) {
          return _humanizeError(error);
        }
      }
    } catch (_) {}

    if (response.statusCode >= 500) {
      return 'На сервере произошла внутренняя ошибка. Повторите попытку позже.';
    }

    return 'Не удалось выполнить анализ фотографии.';
  }

  String _humanizeError(String message) {
    final raw = message.replaceFirst('Exception: ', '').trim();

    if (raw.contains('Часть обязательных ключевых точек') ||
        raw.contains('Требуется фронтальная фотография')) {
      return 'Не удалось надежно определить ключевые точки тела. '
          'Загрузите фронтальную фотографию в полный рост, где полностью видны '
          'плечи, таз, колени и голеностоп.';
    }

    if (raw.contains('SocketException') ||
        raw.contains('Connection refused') ||
        raw.contains('Failed host lookup')) {
      return 'Не удалось подключиться к серверу анализа. '
          'Проверьте, что backend запущен и доступен.';
    }

    return raw;
  }
  Color _qualityColor(BuildContext context, double score) {
    if (score >= 0.85) return Colors.green.shade700;
    if (score >= 0.70) return Colors.teal.shade700;
    if (score >= 0.55) return Colors.orange.shade700;
    return Theme.of(context).colorScheme.error;
  }

  Widget _buildInfoChip({
    required BuildContext context,
    required IconData icon,
    required String label,
    required String value,
    Color? color,
  }) {
    final chipColor = color ?? Theme.of(context).colorScheme.primary;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: chipColor.withValues(alpha: 0.10),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: chipColor.withValues(alpha: 0.18)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 18, color: chipColor),
          const SizedBox(width: 8),
          Flexible(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  label,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Colors.black54,
                      ),
                ),
                const SizedBox(height: 2),
                Text(
                  value,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        fontWeight: FontWeight.w700,
                        color: Colors.black87,
                      ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMetricTile({
    required BuildContext context,
    required String label,
    required String value,
    IconData icon = Icons.straighten_rounded,
  }) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.grey.shade50,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: Theme.of(context).colorScheme.outlineVariant,
        ),
      ),
      child: Row(
        children: [
          Icon(icon, size: 18, color: Theme.of(context).colorScheme.primary),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              label,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
          ),
          const SizedBox(width: 8),
          Text(
            value,
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.w700,
                ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionTitle(BuildContext context, String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Text(
        title,
        style: Theme.of(context).textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w700,
            ),
      ),
    );
  }
  Widget _buildErrorCard(String message) {
    final cs = Theme.of(context).colorScheme;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: cs.errorContainer.withValues(alpha: 0.55),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: cs.error.withValues(alpha: 0.20),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: cs.error.withValues(alpha: 0.10),
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  Icons.error_outline_rounded,
                  color: cs.error,
                  size: 22,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Не удалось выполнить анализ',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.w700,
                            color: cs.onSurface,
                          ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      message,
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            height: 1.35,
                            color: cs.onSurface.withValues(alpha: 0.85),
                          ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.55),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Рекомендации к фото',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        fontWeight: FontWeight.w700,
                      ),
                ),
                const SizedBox(height: 8),
                const Text('• Используйте фронтальное фото в полный рост.'),
                const SizedBox(height: 4),
                const Text('• В кадре должны быть видны плечи, таз, колени и голеностоп.'),
                const SizedBox(height: 4),
                const Text('• Не допускайте перекрытия тела руками, одеждой или посторонними объектами.'),
                const SizedBox(height: 4),
                const Text('• Желательно ровное освещение и нейтральный фон.'),
              ],
            ),
          ),
        ],
      ),
    );
  }
  Widget _buildAnalysisResultCard(CVAnalysisResult result) {
    final qualityColor = _qualityColor(context, result.quality.photoQualityScore);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.03),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.analytics_outlined, color: Theme.of(context).colorScheme.primary),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  'Результаты анализа фото',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w700,
                      ),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.green.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Text(
                  'Анализ выполнен',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Colors.green.shade700,
                        fontWeight: FontWeight.w700,
                      ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),

          _buildSectionTitle(context, 'Общий итог'),
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              _buildInfoChip(
                context: context,
                icon: result.quality.bodyDetected ? Icons.check_circle : Icons.cancel,
                label: 'Тело обнаружено',
                value: _yesNo(result.quality.bodyDetected),
                color: result.quality.bodyDetected ? Colors.green.shade700 : Colors.red.shade700,
              ),
              _buildInfoChip(
                context: context,
                icon: result.quality.bodyFullyVisible ? Icons.crop_free : Icons.crop,
                label: 'Полнота кадра',
                value: _yesNo(result.quality.bodyFullyVisible),
                color: result.quality.bodyFullyVisible ? Colors.teal.shade700 : Colors.orange.shade700,
              ),
              _buildInfoChip(
                context: context,
                icon: Icons.insert_photo_outlined,
                label: 'Качество фото',
                value: _qualityLabel(result.quality.photoQualityScore),
                color: qualityColor,
              ),
            ],
          ),

          const SizedBox(height: 18),
          _buildSectionTitle(context, 'Качество распознавания'),
          Column(
            children: [
              _buildMetricTile(
                context: context,
                label: 'Интегральная оценка качества',
                value: _asPercent(result.quality.photoQualityScore),
                icon: Icons.verified_outlined,
              ),
              const SizedBox(height: 8),
              _buildMetricTile(
                context: context,
                label: 'Средняя уверенность keypoints',
                value: _asPercent(result.quality.keypointConfidenceMean),
                icon: Icons.center_focus_strong,
              ),
              const SizedBox(height: 8),
              _buildMetricTile(
                context: context,
                label: 'Видимые ключевые точки',
                value: _asPercent(result.quality.visibleKeypointsRatio),
                icon: Icons.accessibility_new_rounded,
              ),
            ],
          ),

          const SizedBox(height: 18),
          _buildSectionTitle(context, 'Геометрические признаки'),
          Column(
            children: [
              _buildMetricTile(
                context: context,
                label: 'Наклон корпуса',
                value: '${result.features.torsoTiltDeg.toStringAsFixed(1)}°',
                icon: Icons.accessibility_rounded,
              ),
              const SizedBox(height: 8),
              _buildMetricTile(
                context: context,
                label: 'Наклон плеч',
                value: '${result.features.shoulderTiltDeg.toStringAsFixed(1)}°',
                icon: Icons.height_rounded,
              ),
              const SizedBox(height: 8),
              _buildMetricTile(
                context: context,
                label: 'Наклон таза',
                value: '${result.features.pelvisTiltDeg.toStringAsFixed(1)}°',
                icon: Icons.straighten_rounded,
              ),
              const SizedBox(height: 8),
              _buildMetricTile(
                context: context,
                label: 'Угол левого колена',
                value: '${result.features.leftKneeAngleDeg.toStringAsFixed(1)}°',
                icon: Icons.timeline_rounded,
              ),
              const SizedBox(height: 8),
              _buildMetricTile(
                context: context,
                label: 'Угол правого колена',
                value: '${result.features.rightKneeAngleDeg.toStringAsFixed(1)}°',
                icon: Icons.timeline_rounded,
              ),
            ],
          ),

          if (result.warnings.isNotEmpty) ...[
            const SizedBox(height: 18),
            _buildSectionTitle(context, 'Предупреждения'),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.orange.withValues(alpha: 0.08),
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: Colors.orange.withValues(alpha: 0.25)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: result.warnings
                    .map(
                      (w) => Padding(
                        padding: const EdgeInsets.only(bottom: 6),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Padding(
                              padding: EdgeInsets.only(top: 2),
                              child: Icon(Icons.warning_amber_rounded, size: 18, color: Colors.orange),
                            ),
                            const SizedBox(width: 8),
                            Expanded(child: Text(w)),
                          ],
                        ),
                      ),
                    )
                    .toList(),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildSurrogateErrorCard(String message) {
    final cs = Theme.of(context).colorScheme;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.orange.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.orange.withValues(alpha: 0.25)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.psychology_alt_outlined, color: Colors.orange, size: 22),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Суррогатная оценка недоступна',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w700,
                        color: cs.onSurface,
                      ),
                ),
                const SizedBox(height: 6),
                Text(
                  message,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        height: 1.35,
                        color: cs.onSurface.withValues(alpha: 0.82),
                      ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSurrogateResultCard(SurrogateResult result) {
    final scoreColor = _loadScoreColor(context, result.interpretation.loadScore);
    final stubMode = result.metadata['stub_mode'] == true;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.03),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.psychology_alt_outlined, color: Theme.of(context).colorScheme.primary),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  'Суррогатная биомеханическая оценка',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w700,
                      ),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: scoreColor.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Text(
                  _surrogateLevelLabel(result.interpretation.level),
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: scoreColor,
                        fontWeight: FontWeight.w700,
                      ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),

          _buildSectionTitle(context, 'Общий итог'),
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              _buildInfoChip(
                context: context,
                icon: Icons.speed_rounded,
                label: 'Индекс нагрузки',
                value: '${result.interpretation.loadScore}/100',
                color: scoreColor,
              ),
              _buildInfoChip(
                context: context,
                icon: Icons.flag_outlined,
                label: 'Цель',
                value: _goalLabel(_goal),
                color: Theme.of(context).colorScheme.primary,
              ),
              _buildInfoChip(
                context: context,
                icon: Icons.fitness_center_outlined,
                label: 'Уровень',
                value: _experienceLabel(_experienceLevel),
                color: Colors.teal.shade700,
              ),
            ],
          ),

          const SizedBox(height: 18),
          _buildSectionTitle(context, 'Прогноз суррогатной модели'),
          Column(
            children: [
              _buildMetricTile(
                context: context,
                label: 'Максимальное перемещение (umax)',
                value: _formatCompact(result.prediction.umax),
                icon: Icons.swap_vert_circle_outlined,
              ),
              const SizedBox(height: 8),
              _buildMetricTile(
                context: context,
                label: 'Энергетический отклик (U)',
                value: _formatCompact(result.prediction.u),
                icon: Icons.bolt_outlined,
              ),
              const SizedBox(height: 8),
              _buildMetricTile(
                context: context,
                label: 'Максимум σvm',
                value: _formatCompact(result.prediction.sigmavmMax),
                icon: Icons.show_chart_rounded,
              ),
              const SizedBox(height: 8),
              _buildMetricTile(
                context: context,
                label: 'Опорная реакция Rx',
                value: _formatCompact(result.prediction.rx),
                icon: Icons.compare_arrows_rounded,
              ),
            ],
          ),

          const SizedBox(height: 18),
          _buildSectionTitle(context, 'Входы суррогата'),
          Column(
            children: [
              _buildMetricTile(
                context: context,
                label: 'Lx',
                value: _formatCompact(result.surrogateInput.lx),
                icon: Icons.straighten_rounded,
              ),
              const SizedBox(height: 8),
              _buildMetricTile(
                context: context,
                label: 'Ly',
                value: _formatCompact(result.surrogateInput.ly),
                icon: Icons.straighten_rounded,
              ),
              const SizedBox(height: 8),
              _buildMetricTile(
                context: context,
                label: 'Lz',
                value: _formatCompact(result.surrogateInput.lz),
                icon: Icons.straighten_rounded,
              ),
              const SizedBox(height: 8),
              _buildMetricTile(
                context: context,
                label: 'E',
                value: _formatCompact(result.surrogateInput.e),
                icon: Icons.layers_outlined,
              ),
              const SizedBox(height: 8),
              _buildMetricTile(
                context: context,
                label: 'ν',
                value: _formatCompact(result.surrogateInput.nu),
                icon: Icons.tune_rounded,
              ),
              const SizedBox(height: 8),
              _buildMetricTile(
                context: context,
                label: 'tx',
                value: _formatCompact(result.surrogateInput.tx),
                icon: Icons.arrow_forward_rounded,
              ),
            ],
          ),

          const SizedBox(height: 18),
          _buildSectionTitle(context, 'Интерпретация'),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: scoreColor.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: scoreColor.withValues(alpha: 0.20)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  result.interpretation.summary,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(height: 1.4),
                ),
                const SizedBox(height: 10),
                Text(
                  'Рекомендация по прогрессии',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        fontWeight: FontWeight.w700,
                      ),
                ),
                const SizedBox(height: 6),
                Text(
                  result.interpretation.progressionHint,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(height: 1.4),
                ),
              ],
            ),
          ),

          if (stubMode || result.warnings.isNotEmpty) ...[
            const SizedBox(height: 18),
            _buildSectionTitle(context, 'Замечания'),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.orange.withValues(alpha: 0.08),
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: Colors.orange.withValues(alpha: 0.25)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (stubMode)
                    const Padding(
                      padding: EdgeInsets.only(bottom: 6),
                      child: Text(
                        'Активен stub-режим: значения рассчитаны без реальных артефактов обученной модели.',
                      ),
                    ),
                  ...result.warnings.map(
                    (w) => Padding(
                      padding: const EdgeInsets.only(bottom: 6),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Padding(
                            padding: EdgeInsets.only(top: 2),
                            child: Icon(Icons.info_outline_rounded, size: 18, color: Colors.orange),
                          ),
                          const SizedBox(width: 8),
                          Expanded(child: Text(w)),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final canProceed = _result != null && _surrogateResult != null && !_loading;

    return Scaffold(
      appBar: AppBar(
        automaticallyImplyLeading: false,
        titleSpacing: 20,
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(
            bottom: Radius.circular(28),
          ),
        ),
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'Инициализация профиля',
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                  ),
            ),
            const SizedBox(height: 2),
            Text(
              'Первый шаг персонализации',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: Colors.white.withValues(alpha: 0.9),
                  ),
            ),
          ],
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: IconButton(
              tooltip: 'Справка',
              icon: const Icon(Icons.help_outline_rounded),
              onPressed: _showUserGuide,
            ),
          ),
        ],
        flexibleSpace: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Theme.of(context).colorScheme.primary,
                Theme.of(context).colorScheme.primary.withValues(alpha: 0.82),
              ],
            ),
            boxShadow: [
              BoxShadow(
                color: Theme.of(context).colorScheme.primary.withValues(alpha: 0.20),
                blurRadius: 18,
                offset: const Offset(0, 8),
              ),
            ],
          ),
        ),
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Builder(
              builder: (context) {
                final cs = Theme.of(context).colorScheme;

                return Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(24),
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [
                        cs.primary,
                        cs.primary.withValues(alpha: 0.86),
                      ],
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: cs.primary.withValues(alpha: 0.22),
                        blurRadius: 18,
                        offset: const Offset(0, 8),
                      ),
                    ],
                  ),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Container(
                        padding: const EdgeInsets.all(10),
                        child: const Icon(
                          Icons.monitor_heart_rounded,
                          color: Colors.white,
                          size: 32,
                        ),
                      ),
                      const SizedBox(width: 14),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Фото для анализа осанки и пропорций',
                              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                                    color: Colors.white,
                                    fontWeight: FontWeight.w700,
                                  ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'Загрузите фотографию в полный рост: человек должен стоять прямо, лицом к камере, '
                              'на снимке должны быть видны плечи, таз, колени и голеностоп.',
                              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                    color: Colors.white.withValues(alpha: 0.95),
                                    height: 1.35,
                                  ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
            const SizedBox(height: 12),

            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Form(
                  key: _formKey,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      DropdownButtonFormField<String>(
                        initialValue: _gender,
                        items: const [
                          DropdownMenuItem(value: 'female', child: Text('Женский (female)')),
                          DropdownMenuItem(value: 'male', child: Text('Мужской (male)')),
                        ],
                        onChanged: (v) => setState(() => _gender = v ?? 'female'),
                        decoration: const InputDecoration(
                          labelText: 'Пол',
                          border: OutlineInputBorder(),
                        ),
                      ),
                      const SizedBox(height: 12),

                      TextFormField(
                        controller: _ageController,
                        keyboardType: TextInputType.number,
                        decoration: const InputDecoration(
                          labelText: 'Возраст',
                          hintText: 'Например: 26',
                          border: OutlineInputBorder(),
                        ),
                        validator: (v) {
                          final s = (v ?? '').trim();
                          if (s.isEmpty) return 'Укажите возраст.';
                          final age = int.tryParse(s);
                          if (age == null) return 'Возраст должен быть целым числом.';
                          if (age < 10 || age > 100) return 'Возраст вне допустимого диапазона.';
                          return null;
                        },
                      ),
                      const SizedBox(height: 12),

                      TextFormField(
                        controller: _heightController,
                        keyboardType: const TextInputType.numberWithOptions(decimal: true),
                        decoration: const InputDecoration(
                          labelText: 'Рост, см',
                          hintText: 'Например: 170',
                          border: OutlineInputBorder(),
                        ),
                        validator: (v) {
                          final s = (v ?? '').trim().replaceAll(',', '.');
                          if (s.isEmpty) return 'Укажите рост.';
                          final height = double.tryParse(s);
                          if (height == null) return 'Рост должен быть числом.';
                          if (height <= 100 || height > 250) return 'Рост вне допустимого диапазона.';
                          return null;
                        },
                      ),
                      const SizedBox(height: 12),

                      TextFormField(
                        controller: _weightController,
                        keyboardType: const TextInputType.numberWithOptions(decimal: true),
                        decoration: const InputDecoration(
                          labelText: 'Вес, кг',
                          hintText: 'Например: 65',
                          border: OutlineInputBorder(),
                        ),
                        validator: (v) {
                          final s = (v ?? '').trim().replaceAll(',', '.');
                          if (s.isEmpty) return 'Укажите вес.';
                          final weight = double.tryParse(s);
                          if (weight == null) return 'Вес должен быть числом.';
                          if (weight <= 25 || weight > 300) return 'Вес вне допустимого диапазона.';
                          return null;
                        },
                      ),
                      const SizedBox(height: 12),

                      DropdownButtonFormField<String>(
                        initialValue: _goal,
                        items: const [
                          DropdownMenuItem(
                            value: 'muscle_gain',
                            child: Text('Набор мышечной массы'),
                          ),
                          DropdownMenuItem(
                            value: 'fat_loss',
                            child: Text('Снижение жировой массы'),
                          ),
                          DropdownMenuItem(
                            value: 'maintenance',
                            child: Text('Поддержание формы'),
                          ),
                          DropdownMenuItem(
                            value: 'recomposition',
                            child: Text('Рекомпозиция'),
                          ),
                        ],
                        onChanged: (v) => setState(() => _goal = v ?? 'muscle_gain'),
                        decoration: const InputDecoration(
                          labelText: 'Цель',
                          border: OutlineInputBorder(),
                        ),
                      ),
                      const SizedBox(height: 12),

                      DropdownButtonFormField<String>(
                        initialValue: _experienceLevel,
                        items: const [
                          DropdownMenuItem(
                            value: 'beginner',
                            child: Text('Начальный'),
                          ),
                          DropdownMenuItem(
                            value: 'intermediate',
                            child: Text('Средний'),
                          ),
                          DropdownMenuItem(
                            value: 'advanced',
                            child: Text('Продвинутый'),
                          ),
                        ],
                        onChanged: (v) => setState(() => _experienceLevel = v ?? 'beginner'),
                        decoration: const InputDecoration(
                          labelText: 'Уровень подготовки',
                          border: OutlineInputBorder(),
                        ),
                      ),
                      const SizedBox(height: 12),

                      if (_imageFile == null)
                        OutlinedButton.icon(
                          onPressed: _loading ? null : _pickImage,
                          icon: const Icon(Icons.photo),
                          label: const Text('Выбрать фото'),
                        )
                      else
                        Row(
                          children: [
                            Expanded(
                              child: OutlinedButton.icon(
                                onPressed: _loading ? null : _pickImage,
                                icon: const Icon(Icons.photo),
                                label: const Text('Изменить фото'),
                              ),
                            ),
                            const SizedBox(width: 12),
                            OutlinedButton.icon(
                              onPressed: _loading ? null : _removeImage,
                              icon: const Icon(Icons.delete_outline),
                              label: const Text('Удалить'),
                              style: OutlinedButton.styleFrom(
                                foregroundColor: Theme.of(context).colorScheme.error,
                              ),
                            ),
                          ],
                        ),

                      if (_imageFile != null) ...[
                        const SizedBox(height: 12),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(12),
                          child: Container(
                            width: double.infinity,
                            constraints: const BoxConstraints(
                              minHeight: 220,
                              maxHeight: 420,
                            ),
                            color: Colors.black.withValues(alpha: 0.04),
                            child: Image.file(
                              _imageFile!,
                              fit: BoxFit.contain,
                            ),
                          ),
                        ),
                      ],

                      const SizedBox(height: 16),

                      FilledButton(
                        onPressed: _loading ? null : _onSubmit,
                        child: _loading
                            ? const SizedBox(
                                height: 18,
                                width: 18,
                                child: CircularProgressIndicator(strokeWidth: 2),
                              )
                            : const Text('Проанализировать фото'),
                      ),

                      if (_error != null) ...[
                        const SizedBox(height: 12),
                        _buildErrorCard(_error!),
                      ],

                      if (_result != null) ...[
                        const SizedBox(height: 16),
                        _buildAnalysisResultCard(_result!),
                      ],

                      if (_surrogateError != null) ...[
                        const SizedBox(height: 12),
                        _buildSurrogateErrorCard(_surrogateError!),
                      ],

                      if (_surrogateResult != null) ...[
                        const SizedBox(height: 12),
                        _buildSurrogateResultCard(_surrogateResult!),
                        const SizedBox(height: 12),
                        FilledButton.tonal(
                          onPressed: canProceed ? _goNext : null,
                          child: const Text('Перейти к плану'),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class PlanScreenStub extends StatelessWidget {
  final CVAnalysisResult result;
  final SurrogateResult? surrogateResult;

  const PlanScreenStub({
    super.key,
    required this.result,
    required this.surrogateResult,
  });

  @override
  Widget build(BuildContext context) {
    final surrogate = surrogateResult;

    return Scaffold(
      appBar: AppBar(title: const Text('План (заглушка)')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Text(
          'Далее предполагается генерация персонализированного плана тренировок и питания.\n\n'
          'CV-признаки:\n'
          '- photo_quality_score: ${result.quality.photoQualityScore.toStringAsFixed(3)}\n'
          '- keypoint_confidence_mean: ${result.quality.keypointConfidenceMean.toStringAsFixed(3)}\n'
          '- torso_tilt_deg: ${result.features.torsoTiltDeg.toStringAsFixed(2)}\n'
          '- shoulder_tilt_deg: ${result.features.shoulderTiltDeg.toStringAsFixed(2)}\n'
          '- pelvis_tilt_deg: ${result.features.pelvisTiltDeg.toStringAsFixed(2)}\n\n'
          'Суррогатная оценка:\n'
          '- load_score: ${surrogate?.interpretation.loadScore ?? 0}\n'
          '- level: ${surrogate?.interpretation.level ?? '-'}\n'
          '- umax: ${surrogate != null ? surrogate.prediction.umax.toStringAsFixed(4) : '-'}\n'
          '- U: ${surrogate != null ? surrogate.prediction.u.toStringAsFixed(4) : '-'}\n'
          '- sigmavm_max: ${surrogate != null ? surrogate.prediction.sigmavmMax.toStringAsFixed(4) : '-'}',
        ),
      ),
    );
  }
}
