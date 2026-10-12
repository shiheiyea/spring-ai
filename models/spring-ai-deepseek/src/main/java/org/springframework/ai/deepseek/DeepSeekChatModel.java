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

package org.springframework.ai.deepseek;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import io.micrometer.observation.Observation;
import io.micrometer.observation.ObservationRegistry;
import io.micrometer.observation.contextpropagation.ObservationThreadLocalAccessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.metadata.ChatGenerationMetadata;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.metadata.DefaultUsage;
import org.springframework.ai.chat.metadata.EmptyUsage;
import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.model.MessageAggregator;
import org.springframework.ai.chat.model.StreamingChatModel;
import org.springframework.ai.chat.observation.ChatModelObservationContext;
import org.springframework.ai.chat.observation.ChatModelObservationConvention;
import org.springframework.ai.chat.observation.ChatModelObservationDocumentation;
import org.springframework.ai.chat.observation.DefaultChatModelObservationConvention;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.deepseek.api.DeepSeekApi;
import org.springframework.ai.deepseek.api.DeepSeekApi.ChatCompletion;
import org.springframework.ai.deepseek.api.DeepSeekApi.ChatCompletion.Choice;
import org.springframework.ai.deepseek.api.DeepSeekApi.ChatCompletionMessage;
import org.springframework.ai.deepseek.api.DeepSeekApi.ChatCompletionMessage.ChatCompletionFunction;
import org.springframework.ai.deepseek.api.DeepSeekApi.ChatCompletionMessage.ToolCall;
import org.springframework.ai.deepseek.api.DeepSeekApi.ChatCompletionRequest;
import org.springframework.ai.deepseek.api.common.DeepSeekConstants;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.model.tool.DefaultToolExecutionEligibilityPredicate;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.model.tool.ToolExecutionEligibilityPredicate;
import org.springframework.ai.model.tool.ToolExecutionResult;
import org.springframework.ai.model.tool.internal.ToolCallReactiveContextHolder;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.ai.support.UsageCalculator;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.http.ResponseEntity;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;

/**
 * {@link ChatModel} and {@link StreamingChatModel} implementation for {@literal DeepSeek}
 * backed by {@link DeepSeekApi}.
 *
 * @author Geng Rong
 */
public class DeepSeekChatModel implements ChatModel {

	private static final Logger logger = LoggerFactory.getLogger(DeepSeekChatModel.class);

	private static final ChatModelObservationConvention DEFAULT_OBSERVATION_CONVENTION = new DefaultChatModelObservationConvention();

	private static final ToolCallingManager DEFAULT_TOOL_CALLING_MANAGER = ToolCallingManager.builder().build();

	/**
	 * The default options used for the chat completion requests.
	 */
	private final DeepSeekChatOptions defaultOptions;

	/**
	 * The retry template used to retry the DeepSeek API calls.
	 */
	public final RetryTemplate retryTemplate;

	/**
	 * Low-level access to the DeepSeek API.
	 */
	private final DeepSeekApi deepSeekApi;

	/**
	 * Observation registry used for instrumentation.
	 */
	private final ObservationRegistry observationRegistry;

	/**
	 * The tool calling manager used to execute tools.
	 */
	private final ToolCallingManager toolCallingManager;

	/**
	 * The tool execution eligibility predicate used to determine if a tool can be
	 * executed.
	 */
	private final ToolExecutionEligibilityPredicate toolExecutionEligibilityPredicate;

	/**
	 * Conventions to use for generating observations.
	 */
	private ChatModelObservationConvention observationConvention = DEFAULT_OBSERVATION_CONVENTION;

	public DeepSeekChatModel(DeepSeekApi deepSeekApi, DeepSeekChatOptions defaultOptions,
			ToolCallingManager toolCallingManager, RetryTemplate retryTemplate,
			ObservationRegistry observationRegistry) {
		this(deepSeekApi, defaultOptions, toolCallingManager, retryTemplate, observationRegistry,
				new DefaultToolExecutionEligibilityPredicate());
	}

	public DeepSeekChatModel(DeepSeekApi deepSeekApi, DeepSeekChatOptions defaultOptions,
			ToolCallingManager toolCallingManager, RetryTemplate retryTemplate, ObservationRegistry observationRegistry,
			ToolExecutionEligibilityPredicate toolExecutionEligibilityPredicate) {
		Assert.notNull(deepSeekApi, "deepSeekApi cannot be null");
		Assert.notNull(defaultOptions, "defaultOptions cannot be null");
		Assert.notNull(toolCallingManager, "toolCallingManager cannot be null");
		Assert.notNull(retryTemplate, "retryTemplate cannot be null");
		Assert.notNull(observationRegistry, "observationRegistry cannot be null");
		Assert.notNull(toolExecutionEligibilityPredicate, "toolExecutionEligibilityPredicate cannot be null");
		this.deepSeekApi = deepSeekApi;
		this.defaultOptions = defaultOptions;
		this.toolCallingManager = toolCallingManager;
		this.retryTemplate = retryTemplate;
		this.observationRegistry = observationRegistry;
		this.toolExecutionEligibilityPredicate = toolExecutionEligibilityPredicate;
	}

	@Override
	public ChatResponse call(Prompt prompt) {
		Prompt requestPrompt = buildRequestPrompt(prompt);
		return this.internalCall(requestPrompt, null);
	}

	public ChatResponse internalCall(Prompt prompt, ChatResponse previousChatResponse) {

		ChatCompletionRequest request = createRequest(prompt, false);

		ChatModelObservationContext observationContext = ChatModelObservationContext.builder()
			.prompt(prompt)
			.provider(DeepSeekConstants.PROVIDER_NAME)
			.build();

		ChatResponse response = ChatModelObservationDocumentation.CHAT_MODEL_OPERATION
			.observation(this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry)
			.observe(() -> {

				ResponseEntity<ChatCompletion> completionEntity = this.retryTemplate
					.execute(ctx -> this.deepSeekApi.chatCompletionEntity(request));

				var chatCompletion = completionEntity.getBody();

				if (chatCompletion == null) {
					logger.warn("No chat completion returned for prompt: {}", prompt);
					return new ChatResponse(List.of());
				}

				List<Choice> choices = chatCompletion.choices();
				if (choices == null) {
					logger.warn("No choices returned for prompt: {}", prompt);
					return new ChatResponse(List.of());
				}

				List<Generation> generations = choices.stream().map(choice -> {
			// @formatter:off
					Map<String, Object> metadata = Map.of(
							"id", chatCompletion.id() != null ? chatCompletion.id() : "",
							"role", choice.message().role() != null ? choice.message().role().name() : "",
							"index", choice.index(),
							"finishReason", choice.finishReason() != null ? choice.finishReason().name() : "");
					// @formatter:on
					return buildGeneration(choice, metadata);
				}).toList();

				// Current usage
				DeepSeekApi.Usage usage = completionEntity.getBody().usage();
				Usage currentChatResponseUsage = usage != null ? getDefaultUsage(usage) : new EmptyUsage();
				Usage accumulatedUsage = UsageCalculator.getCumulativeUsage(currentChatResponseUsage,
						previousChatResponse);
				ChatResponse chatResponse = new ChatResponse(generations,
						from(completionEntity.getBody(), accumulatedUsage));

				observationContext.setResponse(chatResponse);

				return chatResponse;

			});

		if (this.toolExecutionEligibilityPredicate.isToolExecutionRequired(prompt.getOptions(), response)) {
			var toolExecutionResult = this.toolCallingManager.executeToolCalls(prompt, response);
			if (toolExecutionResult.returnDirect()) {
				// Return tool execution result directly to the client.
				return ChatResponse.builder()
					.from(response)
					.generations(ToolExecutionResult.buildGenerations(toolExecutionResult))
					.build();
			}
			else {
				// Send the tool execution result back to the model.
				return this.internalCall(new Prompt(toolExecutionResult.conversationHistory(), prompt.getOptions()),
						response);
			}
		}

		return response;
	}

	@Override
	public Flux<ChatResponse> stream(Prompt prompt) {
		Prompt requestPrompt = buildRequestPrompt(prompt);
		return internalStream(requestPrompt, null);
	}

	public Flux<ChatResponse> internalStream(Prompt prompt, ChatResponse previousChatResponse) {
		return Flux.deferContextual(contextView -> {
			// 根据 prompt 构建发送给 DeepSeek 的请求体，第二个参数 true 代表启用 SSE 流式传输 (stream: true)
			ChatCompletionRequest request = createRequest(prompt, true);

			// 调用底层 HTTP 客户端发起 POST 请求，返回一个持续吐出 chunk 数据块的 Flux 响应式流
			Flux<DeepSeekApi.ChatCompletionChunk> completionChunks = this.deepSeekApi.chatCompletionStream(request);

			// 线程安全的哈希表，用于缓存并共享响应块的角色（解决流式下后续 chunk 的 role 字段为 null 的问题）
			// For chunked responses, only the first chunk contains the choice role.
			// The rest of the chunks with same ID share the same role.
			ConcurrentHashMap<String, String> roleMap = new ConcurrentHashMap<>();

			// 构建 Micrometer Observation 观测上下文，记录当前模型供应商（DeepSeek）和初始 Prompt 元数据
			final ChatModelObservationContext observationContext = ChatModelObservationContext.builder()
					.prompt(prompt)
					.provider(DeepSeekConstants.PROVIDER_NAME)
					.build();

			// 创建 Observation 监控实例，用于打点采集吞吐、延迟、错误率等指标
			Observation observation = ChatModelObservationDocumentation.CHAT_MODEL_OPERATION.observation(
					this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry);

			// 从 Reactor 响应式上下文中提取父级 Observation（如果有），建立父子 Span 树状追踪关系，并启动耗时监控计时
			observation.parentObservation(contextView.getOrDefault(ObservationThreadLocalAccessor.KEY, null)).start();

			// 对底层 API 返回的原始 Chunk 流进行处理：先将每个 Chunk 转换为标准的 ChatCompletion 对象
			Flux<ChatResponse> chatResponse = completionChunks.map(this::chunkToChatCompletion)
					// switchMap：将每个 completion 对象包裹为 Mono，并在转换时处理元数据与异常
					.switchMap(chatCompletion -> Mono.just(chatCompletion).map(chatCompletion2 -> {
						try {
							// 获取当前数据块的全局唯一 ID（同一轮对话流的所有 chunk 共享同一个 ID）
							String id = chatCompletion2.id();

							// 遍历模型给出的分支选项 choices，将其封装为 Spring AI 统一的 Generation 生成结果列表
							List<Generation> generations = chatCompletion2.choices().stream().map(choice -> {
								// SSE 协议中通常只有首个 chunk 带有 role（如 "assistant"），后续 chunk 的 role 为 null
								if (choice.message().role() != null) {
									// 命中首个 chunk 时，将 role 登记到 roleMap 中
									roleMap.putIfAbsent(id, choice.message().role().name());
								}

								// @formatter:off
								// 构造当前 chunk 的元数据：带上响应 ID、从 roleMap 回查的角色名、以及完成状态（如 STOP、TOOL_CALLS）
								Map<String, Object> metadata = Map.of(
										"id", chatCompletion2.id(),
										"role", roleMap.getOrDefault(id, ""),
										"finishReason", choice.finishReason() != null ? choice.finishReason().name() : ""
								);
								// @formatter:on
								// 结合原始 choice 数据与 metadata，构造成 Spring AI 规范的 Generation 实例
								return buildGeneration(choice, metadata);
							}).toList();

							// 提取模型返回的 Token 用量（通常仅流的最后一个 chunk 会包含 usage 字段）
							DeepSeekApi.Usage usage = chatCompletion2.usage();
							// 若当前 chunk 有 usage 则解析，否则用空的 EmptyUsage 占位
							Usage currentUsage = (usage != null) ? getDefaultUsage(usage) : new EmptyUsage();
							// 将当前 chunk 消耗与上一轮工具调用的累积 Token 合并累加，得出全局真实的 token 消耗
							Usage cumulativeUsage = UsageCalculator.getCumulativeUsage(currentUsage, previousChatResponse);

							// 组装并返回封装好的 ChatResponse
							return new ChatResponse(generations, from(chatCompletion2, cumulativeUsage));
						}
						catch (Exception e) {
							// 捕获并记录单个 chunk 转换过程中的异常，降级返回一个包含空列表的 ChatResponse 防止流被硬中断
							logger.error("Error processing chat completion", e);
							return new ChatResponse(List.of());
						}

					}));

			// @formatter:off
			// 响应式扁平化映射：针对流中吐出的每个 ChatResponse 检查是否需要执行外部工具（Function Calling）
			Flux<ChatResponse> flux = chatResponse.flatMap(response -> {
						// 断言判断：根据当前用户设定的选项以及响应内容（例如是否存在 tool_calls），判断是否需触发本地工具执行
						if (this.toolExecutionEligibilityPredicate.isToolExecutionRequired(prompt.getOptions(), response)) {
							// FIXME 说明：由于当前大部分 Java 工具实现是阻塞同步代码，需要切换到弹性线程池避免阻塞 Netty
							// 再次使用 deferContextual 以便捕获并向下游工具传递上下文
							// FIXME: bounded elastic needs to be used since tool calling
							//  is currently only synchronous
							return Flux.deferContextual(ctx -> {
								ToolExecutionResult toolExecutionResult;
								try {
									// 将 Reactor 上下文存入 ThreadLocal 桥接器，使工具内部方法（如安全凭据、租户上下文）能获取到参数
									ToolCallReactiveContextHolder.setContext(ctx);
									// 同步反射执行 @Tool 方法（如数据库查询、外部 HTTP 调用），返回执行结果
									toolExecutionResult = this.toolCallingManager.executeToolCalls(prompt, response);
								}
								finally {
									// 清理 ThreadLocal，防止线程复用带来的内存泄漏与上下文串话
									ToolCallReactiveContextHolder.clearContext();
								}
								// 分支 1：如果工具被配置为“直接返回”（returnDirect = true），无需大模型二次组织语言
								if (toolExecutionResult.returnDirect()) {
									// Return tool execution result directly to the client.
									// 将工具执行结果构造成 Generation，直接封装为 ChatResponse 输出给下游客户端
									return Flux.just(ChatResponse.builder().from(response)
											.generations(ToolExecutionResult.buildGenerations(toolExecutionResult))
											.build());
								}
								// 分支 2：经典工具循环 —— 将工具的执行返回值作为 ToolMessage 放入历史记录，再次发起流式请求让模型总结
								else {
									// 递归调用自身 internalStream，带入新的对话上下文并传递当前 response 作为前序 response 供 token 累加
									// Send the tool execution result back to the model.
									return this.internalStream(new Prompt(toolExecutionResult.conversationHistory(), prompt.getOptions()),
											response);
								}

								// 将阻塞的工具执行逻辑切换至 boundedElastic（弹性阻塞线程池），杜绝占用 Reactor 核心 EventLoop 线程
							}).subscribeOn(Schedulers.boundedElastic());
						}
						else {
							// 不需要执行工具，直接将模型的文本输出 chunk 正常向下游流式分发
							return Flux.just(response);
						}
					})
					// 流发生异常时，向 Micrometer Observation 记录异常信息
					.doOnError(observation::error)
					// 无论流正常结束 (onComplete) 还是异常/被客户端取消 (cancel)，都停止观测指标计时并提交 Trace Span
					.doFinally(s -> observation.stop())
					// 将当前 observation 对象写入 Reactor 上下文，供链路中下游的其他反应式算子共享追踪信息
					.contextWrite(ctx -> ctx.put(ObservationThreadLocalAccessor.KEY, observation));
			// @formatter:on

			// 使用 MessageAggregator 对流中的碎片进行聚合：
			// 既保证下游能继续实时收到流式 chunk，又在后台聚合成完整的全量响应传给 observationContext 记录监控日志
			return new MessageAggregator().aggregate(flux, observationContext::setResponse);

		});
	}

	private Generation buildGeneration(Choice choice, Map<String, Object> metadata) {
		List<AssistantMessage.ToolCall> toolCalls = choice.message().toolCalls() == null ? List.of()
				: choice.message()
					.toolCalls()
					.stream()
					.map(toolCall -> new AssistantMessage.ToolCall(toolCall.id(), "function",
							toolCall.function().name(), toolCall.function().arguments()))
					.toList();

		String finishReason = (choice.finishReason() != null ? choice.finishReason().name() : "");
		var generationMetadataBuilder = ChatGenerationMetadata.builder().finishReason(finishReason);

		String textContent = choice.message().content();
		String reasoningContent = choice.message().reasoningContent();

		DeepSeekAssistantMessage.Builder builder = new DeepSeekAssistantMessage.Builder();
		DeepSeekAssistantMessage assistantMessage = builder.content(textContent)
			.reasoningContent(reasoningContent)
			.properties(metadata)
			.toolCalls(toolCalls)
			.build();

		return new Generation(assistantMessage, generationMetadataBuilder.build());
	}

	private ChatResponseMetadata from(DeepSeekApi.ChatCompletion result, Usage usage) {
		Assert.notNull(result, "DeepSeek ChatCompletionResult must not be null");
		var builder = ChatResponseMetadata.builder()
			.id(result.id() != null ? result.id() : "")
			.usage(usage)
			.model(result.model() != null ? result.model() : "")
			.keyValue("created", result.created() != null ? result.created() : 0L)
			.keyValue("system-fingerprint", result.systemFingerprint() != null ? result.systemFingerprint() : "");
		return builder.build();
	}

	private ChatResponseMetadata from(ChatResponseMetadata chatResponseMetadata, Usage usage) {
		Assert.notNull(chatResponseMetadata, "DeepSeek ChatResponseMetadata must not be null");
		var builder = ChatResponseMetadata.builder()
			.id(chatResponseMetadata.getId() != null ? chatResponseMetadata.getId() : "")
			.usage(usage)
			.model(chatResponseMetadata.getModel() != null ? chatResponseMetadata.getModel() : "");
		return builder.build();
	}

	/**
	 * Convert the ChatCompletionChunk into a ChatCompletion. The Usage is set to null.
	 * @param chunk the ChatCompletionChunk to convert
	 * @return the ChatCompletion
	 */
	private DeepSeekApi.ChatCompletion chunkToChatCompletion(DeepSeekApi.ChatCompletionChunk chunk) {
		List<Choice> choices = chunk.choices()
			.stream()
			.map(chunkChoice -> new Choice(chunkChoice.finishReason(), chunkChoice.index(), chunkChoice.delta(),
					chunkChoice.logprobs()))
			.toList();

		return new DeepSeekApi.ChatCompletion(chunk.id(), choices, chunk.created(), chunk.model(), chunk.serviceTier(),
				chunk.systemFingerprint(), chunk.usage());
	}

	private DefaultUsage getDefaultUsage(DeepSeekApi.Usage usage) {
		return new DefaultUsage(usage.promptTokens(), usage.completionTokens(), usage.totalTokens(), usage);
	}

	Prompt buildRequestPrompt(Prompt prompt) {
		DeepSeekChatOptions runtimeOptions = null;
		if (prompt.getOptions() != null) {
			if (prompt.getOptions() instanceof ToolCallingChatOptions toolCallingChatOptions) {
				runtimeOptions = ModelOptionsUtils.copyToTarget(toolCallingChatOptions, ToolCallingChatOptions.class,
						DeepSeekChatOptions.class);
			}
			else {
				runtimeOptions = ModelOptionsUtils.copyToTarget(prompt.getOptions(), ChatOptions.class,
						DeepSeekChatOptions.class);
			}
		}

		DeepSeekChatOptions requestOptions = ModelOptionsUtils.merge(runtimeOptions, this.defaultOptions,
				DeepSeekChatOptions.class);

		if (runtimeOptions != null) {
			requestOptions.setInternalToolExecutionEnabled(
					ModelOptionsUtils.mergeOption(runtimeOptions.getInternalToolExecutionEnabled(),
							this.defaultOptions.getInternalToolExecutionEnabled()));
			requestOptions.setToolNames(ToolCallingChatOptions.mergeToolNames(runtimeOptions.getToolNames(),
					this.defaultOptions.getToolNames()));
			requestOptions.setToolCallbacks(ToolCallingChatOptions.mergeToolCallbacks(runtimeOptions.getToolCallbacks(),
					this.defaultOptions.getToolCallbacks()));
			requestOptions.setToolContext(ToolCallingChatOptions.mergeToolContext(runtimeOptions.getToolContext(),
					this.defaultOptions.getToolContext()));
		}
		else {
			requestOptions.setInternalToolExecutionEnabled(this.defaultOptions.getInternalToolExecutionEnabled());
			requestOptions.setToolNames(this.defaultOptions.getToolNames());
			requestOptions.setToolCallbacks(this.defaultOptions.getToolCallbacks());
			requestOptions.setToolContext(this.defaultOptions.getToolContext());
		}

		ToolCallingChatOptions.validateToolCallbacks(requestOptions.getToolCallbacks());

		return new Prompt(prompt.getInstructions(), requestOptions);
	}

	/**
	 * Accessible for testing.
	 */
	ChatCompletionRequest createRequest(Prompt prompt, boolean stream) {
		List<ChatCompletionMessage> chatCompletionMessages = prompt.getInstructions().stream().map(message -> {
			if (message.getMessageType() == MessageType.USER || message.getMessageType() == MessageType.SYSTEM) {
				return List.of(new ChatCompletionMessage(message.getText(),
						ChatCompletionMessage.Role.valueOf(message.getMessageType().name())));
			}
			else if (message.getMessageType() == MessageType.ASSISTANT) {
				var assistantMessage = (AssistantMessage) message;
				List<ToolCall> toolCalls = null;
				if (!CollectionUtils.isEmpty(assistantMessage.getToolCalls())) {
					toolCalls = assistantMessage.getToolCalls().stream().map(toolCall -> {
						var function = new ChatCompletionFunction(toolCall.name(), toolCall.arguments());
						return new ToolCall(toolCall.id(), toolCall.type(), function);
					}).toList();
				}
				Boolean isPrefixAssistantMessage = null;
				if (message instanceof DeepSeekAssistantMessage
						&& Boolean.TRUE.equals(((DeepSeekAssistantMessage) message).getPrefix())) {
					isPrefixAssistantMessage = true;
				}
				return List.of(new ChatCompletionMessage(assistantMessage.getText(),
						ChatCompletionMessage.Role.ASSISTANT, null, null, toolCalls, isPrefixAssistantMessage, null));
			}
			else if (message.getMessageType() == MessageType.TOOL) {
				ToolResponseMessage toolMessage = (ToolResponseMessage) message;

				toolMessage.getResponses()
					.forEach(response -> Assert.isTrue(response.id() != null, "ToolResponseMessage must have an id"));
				return toolMessage.getResponses()
					.stream()
					.map(tr -> new ChatCompletionMessage(tr.responseData(), ChatCompletionMessage.Role.TOOL, tr.name(),
							tr.id(), null))
					.toList();
			}
			else {
				throw new IllegalArgumentException("Unsupported message type: " + message.getMessageType());
			}
		}).flatMap(List::stream).toList();

		ChatCompletionRequest request = new ChatCompletionRequest(chatCompletionMessages, stream);

		DeepSeekChatOptions requestOptions = (DeepSeekChatOptions) prompt.getOptions();
		request = ModelOptionsUtils.merge(requestOptions, request, ChatCompletionRequest.class);

		// Add the tool definitions to the request's tools parameter.
		List<ToolDefinition> toolDefinitions = this.toolCallingManager.resolveToolDefinitions(requestOptions);
		if (!CollectionUtils.isEmpty(toolDefinitions)) {
			request = ModelOptionsUtils.merge(
					DeepSeekChatOptions.builder().tools(this.getFunctionTools(toolDefinitions)).build(), request,
					ChatCompletionRequest.class);
		}

		return request;
	}

	private List<DeepSeekApi.FunctionTool> getFunctionTools(List<ToolDefinition> toolDefinitions) {
		return toolDefinitions.stream().map(toolDefinition -> {
			var function = new DeepSeekApi.FunctionTool.Function(toolDefinition.description(), toolDefinition.name(),
					toolDefinition.inputSchema());
			return new DeepSeekApi.FunctionTool(function);
		}).toList();
	}

	private ChatOptions buildRequestOptions(DeepSeekApi.ChatCompletionRequest request) {
		return ChatOptions.builder()
			.model(request.model())
			.frequencyPenalty(request.frequencyPenalty())
			.maxTokens(request.maxTokens())
			.presencePenalty(request.presencePenalty())
			.stopSequences(request.stop())
			.temperature(request.temperature())
			.topP(request.topP())
			.build();
	}

	@Override
	public ChatOptions getDefaultOptions() {
		return DeepSeekChatOptions.fromOptions(this.defaultOptions);
	}

	@Override
	public String toString() {
		return "DeepSeekChatModel [defaultOptions=" + this.defaultOptions + "]";
	}

	/**
	 * Use the provided convention for reporting observation data
	 * @param observationConvention The provided convention
	 */
	public void setObservationConvention(ChatModelObservationConvention observationConvention) {
		Assert.notNull(observationConvention, "observationConvention cannot be null");
		this.observationConvention = observationConvention;
	}

	public static Builder builder() {
		return new Builder();
	}

	public static final class Builder {

		private DeepSeekApi deepSeekApi;

		private DeepSeekChatOptions defaultOptions = DeepSeekChatOptions.builder()
			.model(DeepSeekApi.DEFAULT_CHAT_MODEL)
			.temperature(0.7)
			.build();

		private ToolCallingManager toolCallingManager;

		private ToolExecutionEligibilityPredicate toolExecutionEligibilityPredicate = new DefaultToolExecutionEligibilityPredicate();

		private RetryTemplate retryTemplate = RetryUtils.DEFAULT_RETRY_TEMPLATE;

		private ObservationRegistry observationRegistry = ObservationRegistry.NOOP;

		private Builder() {
		}

		public Builder deepSeekApi(DeepSeekApi deepSeekApi) {
			this.deepSeekApi = deepSeekApi;
			return this;
		}

		public Builder defaultOptions(DeepSeekChatOptions defaultOptions) {
			this.defaultOptions = defaultOptions;
			return this;
		}

		public Builder toolCallingManager(ToolCallingManager toolCallingManager) {
			this.toolCallingManager = toolCallingManager;
			return this;
		}

		public Builder toolExecutionEligibilityPredicate(
				ToolExecutionEligibilityPredicate toolExecutionEligibilityPredicate) {
			this.toolExecutionEligibilityPredicate = toolExecutionEligibilityPredicate;
			return this;
		}

		public Builder retryTemplate(RetryTemplate retryTemplate) {
			this.retryTemplate = retryTemplate;
			return this;
		}

		public Builder observationRegistry(ObservationRegistry observationRegistry) {
			this.observationRegistry = observationRegistry;
			return this;
		}

		public DeepSeekChatModel build() {
			if (this.toolCallingManager != null) {
				return new DeepSeekChatModel(this.deepSeekApi, this.defaultOptions, this.toolCallingManager,
						this.retryTemplate, this.observationRegistry, this.toolExecutionEligibilityPredicate);
			}
			return new DeepSeekChatModel(this.deepSeekApi, this.defaultOptions, DEFAULT_TOOL_CALLING_MANAGER,
					this.retryTemplate, this.observationRegistry, this.toolExecutionEligibilityPredicate);
		}

	}

}
