/*
 * Copyright 2023-2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.model;

import java.beans.PropertyDescriptor;
import java.lang.reflect.Field;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.cfg.CoercionAction;
import com.fasterxml.jackson.databind.cfg.CoercionInputShape;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.github.victools.jsonschema.generator.Option;
import com.github.victools.jsonschema.generator.OptionPreset;
import com.github.victools.jsonschema.generator.SchemaGenerator;
import com.github.victools.jsonschema.generator.SchemaGeneratorConfig;
import com.github.victools.jsonschema.generator.SchemaGeneratorConfigBuilder;
import com.github.victools.jsonschema.generator.SchemaVersion;
import com.github.victools.jsonschema.module.jackson.JacksonModule;
import com.github.victools.jsonschema.module.jackson.JacksonOption;
import com.github.victools.jsonschema.module.swagger2.Swagger2Module;

import org.springframework.ai.util.JacksonUtils;
import org.springframework.beans.BeanWrapper;
import org.springframework.beans.BeanWrapperImpl;
import org.springframework.core.KotlinDetector;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.ObjectUtils;

/**
 * Utility class for manipulating {@link ModelOptions} objects.
 *
 * @author Christian Tzolov
 * @author Thomas Vitale
 * @since 0.8.0
 */
public abstract class ModelOptionsUtils {

	public static final ObjectMapper OBJECT_MAPPER = JsonMapper.builder()
			.disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
			.disable(SerializationFeature.FAIL_ON_EMPTY_BEANS)
			.addModules(JacksonUtils.instantiateAvailableModules())
			.build()
			.configure(DeserializationFeature.ACCEPT_EMPTY_STRING_AS_NULL_OBJECT, true);

	static {
		// Configure coercion for empty strings to null for Enum types
		// This fixes the issue where empty string finish_reason values cause
		// deserialization failures
		OBJECT_MAPPER.coercionConfigFor(Enum.class).setCoercion(CoercionInputShape.EmptyString, CoercionAction.AsNull);
	}

	private static final List<String> BEAN_MERGE_FIELD_EXCISIONS = List.of("class");

	private static final ConcurrentHashMap<Class<?>, List<String>> REQUEST_FIELD_NAMES_PER_CLASS = new ConcurrentHashMap<>();

	private static final AtomicReference<SchemaGenerator> SCHEMA_GENERATOR_CACHE = new AtomicReference<>();

	private static TypeReference<HashMap<String, Object>> MAP_TYPE_REF = new TypeReference<>() {

	};

	/**
	 * Converts the given JSON string to a Map of String and Object using the default
	 * ObjectMapper.
	 * @param json the JSON string to convert to a Map.
	 * @return the converted Map.
	 */
	public static Map<String, Object> jsonToMap(String json) {
		return jsonToMap(json, OBJECT_MAPPER);
	}

	/**
	 * Converts the given JSON string to a Map of String and Object using a custom
	 * ObjectMapper.
	 * @param json the JSON string to convert to a Map.
	 * @param objectMapper the ObjectMapper to use for deserialization.
	 * @return the converted Map.
	 */
	public static Map<String, Object> jsonToMap(String json, ObjectMapper objectMapper) {
		try {
			return objectMapper.readValue(json, MAP_TYPE_REF);
		}
		catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * Converts the given JSON string to an Object of the given type.
	 * @param <T> the type of the object to return.
	 * @param json the JSON string to convert to an object.
	 * @param type the type of the object to return.
	 * @return Object instance of the given type.
	 */
	public static <T> T jsonToObject(String json, Class<T> type) {
		try {
			return OBJECT_MAPPER.readValue(json, type);
		}
		catch (Exception e) {
			throw new RuntimeException("Failed to json: " + json, e);
		}
	}

	/**
	 * Converts the given object to a JSON string.
	 * @param object the object to convert to a JSON string.
	 * @return the JSON string.
	 */
	public static String toJsonString(Object object) {
		try {
			return OBJECT_MAPPER.writeValueAsString(object);
		}
		catch (JsonProcessingException e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * Converts the given object to a JSON string.
	 * @param object the object to convert to a JSON string.
	 * @return the JSON string.
	 */
	public static String toJsonStringPrettyPrinter(Object object) {
		try {
			return OBJECT_MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(object);
		}
		catch (JsonProcessingException e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * Merges the source object into the target object and returns an object represented
	 * by the given class. The JSON property names are used to match the fields to merge.
	 * The source non-null values override the target values with the same field name. The
	 * source null values are ignored. If the acceptedFieldNames is not empty, only the
	 * fields with the given names are merged and returned. If the acceptedFieldNames is
	 * empty, use the {@code @JsonProperty} names, inferred from the provided clazz.
	 * @param <T> they type of the class to return.
	 * @param source the source object to merge.
	 * @param target the target object to merge into.
	 * @param clazz the class to return.
	 * @param acceptedFieldNames the list of field names accepted for the target object.
	 * @return the merged object represented by the given class.
	 */
	public static <T> T merge(Object source, Object target, Class<T> clazz, List<String> acceptedFieldNames) {

		if (source == null) {
			source = Map.of();
		}

		// 如果调用方指定了 acceptedFieldNames，直接使用该列表作为允许保留的字段名。
		List<String> requestFieldNames = CollectionUtils.isEmpty(acceptedFieldNames)
				// 通过 REQUEST_FIELD_NAMES_PER_CLASS 缓存池
				// 调用 ModelOptionsUtils::getJsonPropertyValues 反射提取目标类 clazz 上所有标注了 @JsonProperty 的 JSON key 名字（如 "temperature"、"max_tokens"、"top_p" 等）。
				? REQUEST_FIELD_NAMES_PER_CLASS.computeIfAbsent(clazz, ModelOptionsUtils::getJsonPropertyValues)
				: acceptedFieldNames;

		// 利用 Jackson 的 ObjectMapper.convertValue(obj, Map.class) 将源对象和目标对象转换为标准的键值对 Map。
		if (CollectionUtils.isEmpty(requestFieldNames)) {
			throw new IllegalArgumentException("No @JsonProperty fields found in the " + clazz.getName());
		}

		// 利用 Jackson 的 ObjectMapper.convertValue(obj, Map.class) 将源对象和目标对象转换为标准的键值对 Map。
		// 无论入参是 Map、通用的 ChatOptions 接口实现，还是模型专有的 Options 实体，转换后统一变成以 JSON 字段名为 Key 的键值对结构，无需关心具体的 Java 继承层级。
		Map<String, Object> sourceMap = ModelOptionsUtils.objectToMap(source);
		Map<String, Object> targetMap = ModelOptionsUtils.objectToMap(target);

		// 1. 非空覆盖：source 覆盖 target
		targetMap.putAll(sourceMap.entrySet()
				.stream()
				.filter(e -> e.getValue() != null)
				.collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)));

		// 2. 白名单清洗：过滤掉不属于目标类的字段
		targetMap = targetMap.entrySet()
				.stream()
				.filter(e -> requestFieldNames.contains(e.getKey()))
				.collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

		return ModelOptionsUtils.mapToClass(targetMap, clazz);
	}

	/**
	 * Merges the source object into the target object and returns an object represented
	 * by the given class. The JSON property names are used to match the fields to merge.
	 * The source non-null values override the target values with the same field name. The
	 * source null values are ignored. Returns the only field names that match the
	 * {@code @JsonProperty} names, inferred from the provided clazz.
	 * @param <T> they type of the class to return.
	 * @param source the source object to merge.
	 * @param target the target object to merge into.
	 * @param clazz the class to return.
	 * @return the merged object represented by the given class.
	 */
	public static <T> T merge(Object source, Object target, Class<T> clazz) {
		return ModelOptionsUtils.merge(source, target, clazz, null);
	}

	/**
	 * Converts the given object to a Map.
	 * @param source the object to convert to a Map.
	 * @return the converted Map.
	 */
	public static Map<String, Object> objectToMap(Object source) {
		if (source == null) {
			return new HashMap<>();
		}
		try {
			String json = OBJECT_MAPPER.writeValueAsString(source);
			return OBJECT_MAPPER.readValue(json, new TypeReference<Map<String, Object>>() {

					})
					.entrySet()
					.stream()
					.filter(e -> e.getValue() != null)
					.collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
		}
		catch (JsonProcessingException e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * Converts the given Map to the given class.
	 * @param <T> the type of the class to return.
	 * @param source the Map to convert to the given class.
	 * @param clazz the class to convert the Map to.
	 * @return the converted class.
	 */
	public static <T> T mapToClass(Map<String, Object> source, Class<T> clazz) {
		try {
			String json = OBJECT_MAPPER.writeValueAsString(source);
			return OBJECT_MAPPER.readValue(json, clazz);
		}
		catch (JsonProcessingException e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * Returns the list of name values of the {@link JsonProperty} annotations.
	 * @param clazz the class that contains fields annotated with {@link JsonProperty}.
	 * @return the list of values of the {@link JsonProperty} annotations.
	 */
	public static List<String> getJsonPropertyValues(Class<?> clazz) {
		List<String> values = new ArrayList<>();
		Field[] fields = clazz.getDeclaredFields();
		for (Field field : fields) {
			JsonProperty jsonPropertyAnnotation = field.getAnnotation(JsonProperty.class);
			if (jsonPropertyAnnotation != null) {
				values.add(jsonPropertyAnnotation.value());
			}
		}
		return values;
	}

	/**
	 * Returns a new instance of the targetBeanClazz that copies the bean values from the
	 * sourceBean instance.
	 * @param sourceBean the source bean to copy the values from.
	 * @param sourceInterfaceClazz the source interface class. Only the fields with the
	 * same name as the interface methods are copied. This allow the source object to be a
	 * subclass of the source interface with additional, non-interface fields.
	 * @param targetBeanClazz the target class, a subclass of the ChatOptions, to convert
	 * into.
	 * @param <T> the target class type.
	 * @return a new instance of the targetBeanClazz with the values from the sourceBean
	 * instance.
	 */
	public static <I, S extends I, T extends S> T copyToTarget(S sourceBean, Class<I> sourceInterfaceClazz,
															   Class<T> targetBeanClazz) {

		Assert.notNull(sourceInterfaceClazz, "SourceOptionsClazz must not be null");
		Assert.notNull(targetBeanClazz, "TargetOptionsClazz must not be null");

		// 若上层并未配置任何 options（传入 null），直接短路返回 null，避免无谓的实例构建与反射。
		if (sourceBean == null) {
			return null;
		}

		// 检查源对象本身是否已经是目标类型（或目标类型的子类）。如果是，无需发生任何反射和复制，直接强转返回原生对象，降低内存分配与性能开销。
		if (sourceBean.getClass().isAssignableFrom(targetBeanClazz)) {
			return (T) sourceBean;
		}

		try {
			// 通过目标类的无参构造器 targetBeanClazz.getConstructor().newInstance() 反射创建目标对象的新实例。因此，目标类必须提供无参构造函数。
			T targetOptions = targetBeanClazz.getConstructor().newInstance();

			ModelOptionsUtils.mergeBeans(sourceBean, targetOptions, sourceInterfaceClazz, true);

			return targetOptions;
		}
		catch (Exception e) {
			throw new RuntimeException(
					"Failed to convert the " + sourceInterfaceClazz.getName() + " into " + targetBeanClazz.getName(),
					e);
		}
	}

	/**
	 * Merges the source object into the target object. The source null values are
	 * ignored. Only objects with Getter and Setter methods are supported.
	 * @param <T> the type of the source and target object.
	 * @param source the source object to merge.
	 * @param target the target object to merge into.
	 * @param sourceInterfaceClazz the source interface class. Only the fields with the
	 * same name as the interface methods are merged. This allow the source object to be a
	 * subclass of the source interface with additional, non-interface fields.
	 * @param overrideNonNullTargetValues if true, the source non-null values override the
	 * target values with the same field name. If false, the source non-null values are
	 * ignored.
	 * @return the merged target object.
	 */
	public static <I, S extends I, T extends S> T mergeBeans(S source, T target, Class<I> sourceInterfaceClazz,
															 boolean overrideNonNullTargetValues) {
		Assert.notNull(source, "Source object must not be null");
		Assert.notNull(target, "Target object must not be null");

		// 采用 Spring 框架核心的 BeanWrapperImpl 分别包装两个对象。
		// BeanWrapper 提供了标准 JavaBean 的读写抽象（自动处理 getter/setter 匹配及必要的基础类型转换），避免手写原始反射调用。
		BeanWrapper sourceBeanWrap = new BeanWrapperImpl(source);
		BeanWrapper targetBeanWrap = new BeanWrapperImpl(target);

		// 获取传入接口 sourceInterfaceClazz 中定义的所有方法名称列表。
		// 这是实现“接口隔离”的关键。如果 source 是一个包含了 20 个私有扩展属性的具体实现类，
		// 但 sourceInterfaceClazz 接口中只定义了 5 个规范方法，则只有这 5 个方法对应的属性会被纳入合并范围，其余字段直接忽略。
		List<String> interfaceNames = Arrays.stream(sourceInterfaceClazz.getMethods()).map(m -> m.getName()).toList();

		// getPropertyDescriptors() 返回源对象所有的 JavaBean 属性（包含属性名及对应的读写方法）。
		for (PropertyDescriptor descriptor : sourceBeanWrap.getPropertyDescriptors()) {

			// 排除不需要合并的黑名单字段（例如常见的 class 属性，即 getClass() 衍生出的只读属性）。
			// 检查该属性对应的 Getter 方法（通过工具方法 toGetName 将属性名如 model 转换为 getModel）是否存在于前面提取的接口方法列表中。
			// 只有符合接口定义的规范属性才允许放行。
			if (!BEAN_MERGE_FIELD_EXCISIONS.contains(descriptor.getName())
					&& interfaceNames.contains(toGetName(descriptor.getName()))) {

				String propertyName = descriptor.getName();
				Object value = sourceBeanWrap.getPropertyValue(propertyName);

				// Copy value to the target object’
				// 确保源对象中未配置的属性（值为 null）绝不会覆盖或清空目标对象已有的配置。
				if (value != null) {
					var targetValue = targetBeanWrap.getPropertyValue(propertyName);

					if (targetValue == null || overrideNonNullTargetValues) {
						targetBeanWrap.setPropertyValue(propertyName, value);
					}
				}
			}
		}

		return target;
	}

	private static String toGetName(String name) {
		return "get" + name.substring(0, 1).toUpperCase() + name.substring(1);
	}

	/**
	 * Generates JSON Schema (version 2020_12) for the given class.
	 * @param inputType the input {@link Type} to generate JSON Schema from.
	 * @param toUpperCaseTypeValues if true, the type values are converted to upper case.
	 * @return the generated JSON Schema as a String.
	 */
	public static String getJsonSchema(Type inputType, boolean toUpperCaseTypeValues) {

		if (SCHEMA_GENERATOR_CACHE.get() == null) {

			JacksonModule jacksonModule = new JacksonModule(JacksonOption.RESPECT_JSONPROPERTY_REQUIRED);
			Swagger2Module swaggerModule = new Swagger2Module();

			SchemaGeneratorConfigBuilder configBuilder = new SchemaGeneratorConfigBuilder(SchemaVersion.DRAFT_2020_12,
					OptionPreset.PLAIN_JSON)
					.with(Option.EXTRA_OPEN_API_FORMAT_VALUES)
					.with(Option.PLAIN_DEFINITION_KEYS)
					.with(swaggerModule)
					.with(jacksonModule);

			if (KotlinDetector.isKotlinReflectPresent()) {
				configBuilder.with(new KotlinModule());
			}

			SchemaGeneratorConfig config = configBuilder.build();
			SchemaGenerator generator = new SchemaGenerator(config);
			SCHEMA_GENERATOR_CACHE.compareAndSet(null, generator);
		}

		ObjectNode node = SCHEMA_GENERATOR_CACHE.get().generateSchema(inputType);

		if ((inputType == Void.class) && !node.has("properties")) {
			node.putObject("properties");
		}

		if (toUpperCaseTypeValues) { // Required for OpenAPI 3.0 (at least Vertex AI
			// version of it).
			toUpperCaseTypeValues(node);
		}

		return node.toPrettyString();
	}

	public static void toUpperCaseTypeValues(ObjectNode node) {
		if (node == null) {
			return;
		}
		if (node.isObject()) {
			node.fields().forEachRemaining(entry -> {
				JsonNode value = entry.getValue();
				if (value.isObject()) {
					toUpperCaseTypeValues((ObjectNode) value);
				}
				else if (value.isArray()) {
					((ArrayNode) value).elements().forEachRemaining(element -> {
						if (element.isObject() || element.isArray()) {
							toUpperCaseTypeValues((ObjectNode) element);
						}
					});
				}
				else if (value.isTextual() && entry.getKey().equals("type")) {
					String oldValue = ((ObjectNode) node).get("type").asText();
					((ObjectNode) node).put("type", oldValue.toUpperCase());
				}
			});
		}
		else if (node.isArray()) {
			node.elements().forEachRemaining(element -> {
				if (element.isObject() || element.isArray()) {
					toUpperCaseTypeValues((ObjectNode) element);
				}
			});
		}
	}

	/**
	 * Return the runtime value if not empty, or else the default value.
	 */
	public static <T> T mergeOption(T runtimeValue, T defaultValue) {
		return ObjectUtils.isEmpty(runtimeValue) ? defaultValue : runtimeValue;
	}

}
