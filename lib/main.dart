import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fitness ML',
      theme: ThemeData(useMaterial3: true),
      home: const FirstScreen(),
    );
  }
}

class PredictionResult {
  final String bmiCategory;
  final double confidence;

  PredictionResult({required this.bmiCategory, required this.confidence});

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      bmiCategory: (json['bmi_category'] ?? '').toString(),
      confidence: (json['confidence'] is num) ? (json['confidence'] as num).toDouble() : 0.0,
    );
  }
}

class FirstScreen extends StatefulWidget {
  const FirstScreen({super.key});

  @override
  State<FirstScreen> createState() => _FirstScreenState();
}

class _FirstScreenState extends State<FirstScreen> {
  // Важно: endpoint должен совпадать с бекендом /predict (FastAPI).
  // См. app.py: file + gender + age. :contentReference[oaicite:1]{index=1}
  static const String _apiBaseUrl = 'http://10.0.2.2:8000';
  static const String _predictPath = '/predict';

  final _formKey = GlobalKey<FormState>();
  final _ageController = TextEditingController();

  String _gender = 'female'; // ожидается 'female' или 'male'
  File? _imageFile;

  bool _loading = false;
  String? _error;
  PredictionResult? _result;

  final ImagePicker _picker = ImagePicker();

  @override
  void dispose() {
    _ageController.dispose();
    super.dispose();
  }

  Future<void> _pickImage() async {
    setState(() {
      _error = null;
      _result = null;
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

  Future<PredictionResult> _predict({
    required File image,
    required String gender,
    required int age,
  }) async {
    final uri = Uri.parse('$_apiBaseUrl$_predictPath');

    final request = http.MultipartRequest('POST', uri)
      ..fields['gender'] = gender
      ..fields['age'] = age.toString()
      ..files.add(await http.MultipartFile.fromPath('file', image.path));

    final streamed = await request.send();
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode != 200) {
      // Бекенд возвращает {"error": "..."} при исключении. :contentReference[oaicite:2]{index=2}
      throw Exception('HTTP ${response.statusCode}: ${response.body}');
    }

    final Map<String, dynamic> jsonBody = jsonDecode(response.body) as Map<String, dynamic>;
    if (jsonBody.containsKey('error')) {
      throw Exception(jsonBody['error'].toString());
    }

    return PredictionResult.fromJson(jsonBody);
  }

  Future<void> _onSubmit() async {
    setState(() {
      _error = null;
      _result = null;
    });

    if (!_formKey.currentState!.validate()) return;
    if (_imageFile == null) {
      setState(() => _error = 'Не выбрано изображение.');
      return;
    }

    final int age = int.parse(_ageController.text.trim());

    setState(() => _loading = true);
    try {
      final res = await _predict(image: _imageFile!, gender: _gender, age: age);
      setState(() => _result = res);
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _loading = false);
    }
  }

  void _goNext() {
    // Заглушка: переход на экран генерации плана тренировок/диеты.
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => PlanScreenStub(
          bmiCategory: _result?.bmiCategory ?? '',
          confidence: _result?.confidence ?? 0.0,
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
          title: const Text('Руководство пользователя'),
          content: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: const <Widget>[
                Text('1. Инициализация:'),
                Text('Введите базовые параметры и загрузите фото для оценки вашего физического состояния.'),
                SizedBox(height: 10),
                Text('2. Цель:'),
                Text('Выберите цель.'),
                SizedBox(height: 10),
                Text('3. Результаты анализа:'),
                Text('Ознакомтесь с информацией, полученной после анализа ваших параметров.'),
                SizedBox(height: 10),
                Text('4. План тренировок и диет:'),
                Text('Ознакомтесь с составленным планом тренировок и диет.'),
                SizedBox(height: 10),
                Text('5. Прогресс:'),
                Text('Следите за вашим прогрессом к своей цели.'),
              ],
            ),
          ),
          actions: <Widget>[
            TextButton(
              child: const Text('Закрыть'),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final canProceed = _result != null && !_loading;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Инициализация профиля'),
        actions: <Widget>[
          IconButton(
            icon: const Icon(Icons.help_outline),
            onPressed: _showUserGuide, // При нажатии на кнопку отображается руководство
          ),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      child: Text(
                        'Укажите пол, возраст и фото для оценки BMI и персонализации рекомендаций.',
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.info_outline),
                      tooltip: 'Подробнее',
                      onPressed: _showBmiInfo,
                    ),
                  ],
                ),
              ),
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
                        value: _gender,
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
                          if (age < 5 || age > 120) return 'Возраст вне допустимого диапазона.';
                          return null;
                        },
                      ),
                      const SizedBox(height: 12),

                      OutlinedButton.icon(
                        onPressed: _loading ? null : _pickImage,
                        icon: const Icon(Icons.photo),
                        label: Text(_imageFile == null ? 'Выбрать фото' : 'Изменить фото'),
                      ),

                      if (_imageFile != null) ...[
                        const SizedBox(height: 12),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(12),
                          child: Image.file(_imageFile!, height: 220, fit: BoxFit.cover),
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
                            : const Text('Рассчитать категорию BMI'),
                      ),

                      if (_error != null) ...[
                        const SizedBox(height: 12),
                        Text(
                          _error!,
                          style: TextStyle(color: Theme.of(context).colorScheme.error),
                        ),
                      ],

                      if (_result != null) ...[
                        const SizedBox(height: 16),
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('Результат:',
                                  style: Theme.of(context).textTheme.titleMedium),
                              const SizedBox(height: 6),
                              Text('Категория BMI: ${_result!.bmiCategory}'),
                              Text('Достоверность (max prob): ${_result!.confidence.toStringAsFixed(3)}'),
                            ],
                          ),
                        ),
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
  final String bmiCategory;
  final double confidence;

  const PlanScreenStub({
    super.key,
    required this.bmiCategory,
    required this.confidence,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('План (заглушка)')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Text(
          'Далее предполагается генерация персонализированного плана тренировок и питания.\n\n'
          'Входные признаки:\n'
          '- BMI категория: $bmiCategory\n'
          '- confidence: ${confidence.toStringAsFixed(3)}',
        ),
      ),
    );
  }
}
