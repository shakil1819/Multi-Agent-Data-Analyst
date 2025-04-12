import argparse
import asyncio
import os

import gradio as gr
import pandas as pd
import sqlalchemy as sql

from gradio import ChatMessage
from pydantic_ai.usage import UsageLimits

from app.agent_orchestrator import process_user_input_stream, run_agent_orchestrator
from app.visualization_server import serve_visualization


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-Agent Data Analysis Orchestrator")

    parser.add_argument("--prompt", "-p", type=str, help="Analysis prompt or question")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["dataframe", "sql", "auto"],
        default="auto",
        help="Analysis mode: dataframe, sql, or auto-detect",
    )
    parser.add_argument("--file", "-f", type=str, help="Path to data file (.csv, .xlsx)")
    parser.add_argument("--db", "-d", type=str, help="Database connection string")
    parser.add_argument("--stream", "-s", action="store_true", help="Stream output")
    parser.add_argument("--token-limit", type=int, default=4000, help="Maximum total token usage limit")
    parser.add_argument("--request-limit", type=int, default=10, help="Maximum number of requests")

    # For Excel files
    parser.add_argument("--sheet", type=str, help="Sheet name for Excel files")

    return parser.parse_args()


async def run_with_args(args):
    """Run the agent with parsed arguments."""
    if not args.prompt:
        args.prompt = input("Enter your prompt for the data analyst agents: ")

    # Configure usage limits with the correct parameters
    usage_limits = UsageLimits(total_tokens_limit=args.token_limit, request_limit=args.request_limit)

    if args.mode == "sql" or (args.mode == "auto" and args.db):
        # SQL database mode
        if not args.db:
            print("Error: --db parameter is required in SQL mode")
            return {"error": "No database connection specified"}

        if args.stream:
            # Return streaming generator for caller to handle
            return await stream_sql_mode(args, usage_limits)
        # Process normally
        try:
            print(f"Connecting to database: {args.db}")

            # Use the orchestrator to handle everything
            result = await run_agent_orchestrator(user_input=args.prompt, db_url=args.db, usage_limits=usage_limits)

            return result  # noqa: RET504

        except Exception as e:
            print(f"Error connecting to database: {e}")
            return {"error": str(e)}

    elif args.mode == "dataframe" or (args.mode == "auto" and args.file):
        # DataFrame mode
        if not args.file:
            print("Error: --file parameter is required in dataframe mode")
            return {"error": "No file specified"}

        if args.stream:
            # Return streaming generator for caller to handle
            return await stream_dataframe_mode(args, usage_limits)
        # Process normally
        try:
            print(f"Loading data from: {args.file}")

            # Use the orchestrator to handle everything
            result = await run_agent_orchestrator(
                user_input=args.prompt, data_path=args.file, usage_limits=usage_limits
            )

            return result  # noqa: RET504

        except Exception as e:
            print(f"Error loading data: {e}")
            return {"error": str(e)}
    else:
        print("Error: Either --file or --db parameter is required")
        return {"error": "No data source specified"}


async def stream_sql_mode(args, usage_limits):
    """Stream results for SQL mode."""
    try:
        print(f"Connecting to database: {args.db}")
        engine = sql.create_engine(args.db)
        conn = engine.connect()

        # Use streaming version of process_user_input
        async for message in process_user_input_stream(
            user_input=args.prompt, db_connection=conn, usage_limits=usage_limits
        ):
            print(message, end="", flush=True)

        conn.close()
        return {"success": True, "message": "Streaming completed"}

    except Exception as e:
        print(f"Error in SQL streaming mode: {e}")
        return {"error": str(e)}


async def stream_dataframe_mode(args, usage_limits):
    """Stream results for DataFrame mode."""
    try:
        print(f"Loading data from: {args.file}")
        if args.file.endswith(".csv"):
            df = pd.read_csv(args.file)
        elif args.file.endswith((".xls", ".xlsx")):
            df = pd.read_excel(args.file, sheet_name=args.sheet) if args.sheet else pd.read_excel(args.file)
        else:
            print("Unsupported file format. Please use .csv, .xls, or .xlsx")
            return {"error": "Unsupported file format"}

        # Use streaming version of process_user_input
        async for message in process_user_input_stream(user_input=args.prompt, data=df, usage_limits=usage_limits):
            print(message, end="", flush=True)

        return {"success": True, "message": "Streaming completed"}

    except Exception as e:
        print(f"Error in DataFrame streaming mode: {e}")
        return {"error": str(e)}


def display_results(result):
    """Display the results in a user-friendly format."""
    if "error" in result:
        print(f"\n--- Error ---\n{result['error']}")
        return

    print("\n--- Analysis Results ---")

    if result.get("message"):
        print(f"\n{result['message']}")

    if result.get("sql_query"):
        print("\nSQL Query:")
        print(result["sql_query"])

    if result.get("visualization_path"):
        print(f"\nVisualization saved to: {result['visualization_path']}")
        try:
            import webbrowser

            webbrowser.open("file://" + os.path.realpath(result["visualization_path"]))
        except Exception as e:
            print(f"Could not open browser: {e}")

    if result.get("data_summary"):
        print("\nData Summary:")
        shape = result["data_summary"].get("shape")
        if shape:
            print(f"Shape: {shape[0]} rows × {shape[1]} columns")

        columns = result["data_summary"].get("columns")
        if columns:
            print(f"Columns: {', '.join(columns)}")


# Gradio UI implementation
async def process_query(history, prompt, mode, file_upload, db_connection, token_limit, request_limit):  # noqa: C901
    """Process user query and update chat history."""
    if not prompt.strip():
        history.append(ChatMessage(role="assistant", content="Please enter a question or analysis prompt."))
        yield history, gr.update(visible=False)
        return

    # Add user message to history
    history.append(ChatMessage(role="user", content=prompt))
    yield history, gr.update(visible=False)

    # Configure usage limits
    usage_limits = UsageLimits(total_tokens_limit=int(token_limit), request_limit=int(request_limit))

    # Determine the mode
    if mode == "auto":
        if file_upload:
            mode = "dataframe"
        elif db_connection:
            mode = "sql"
        else:
            history.append(
                ChatMessage(
                    role="assistant",
                    content="Please provide either a data file or database connection.",
                    metadata={"title": "❌ Error"},
                )
            )
            yield history, gr.update(visible=False)
            return

    # Process based on mode
    if mode == "sql":
        if not db_connection:
            history.append(
                ChatMessage(
                    role="assistant",
                    content="Database connection string is required for SQL mode.",
                    metadata={"title": "❌ Error"},
                )
            )
            yield history, gr.update(visible=False)
            return

        # Notify user
        history.append(
            ChatMessage(
                role="assistant",
                content=f"Connecting to database: {db_connection}",
                metadata={"title": "🔄 Processing"},
            )
        )
        yield history, gr.update(visible=False)

        try:
            # Stream results
            current_message = ""
            async for message_part in process_user_input_stream(
                user_input=prompt, db_url=db_connection, usage_limits=usage_limits
            ):
                current_message += message_part
                history[-1] = ChatMessage(
                    role="assistant", content=current_message, metadata={"title": "🛠️ Analyzing SQL Data"}
                )
                yield history, gr.update(visible=False)
                await asyncio.sleep(0.01)  # Small delay for smoother streaming

            # Check if visualization was created
            if "visualization saved" in current_message.lower() or "visualization generated" in current_message.lower():
                viz_path = None
                for line in current_message.split("\n"):
                    if "visualization" in line.lower() and ":" in line:
                        viz_path = line.split(":", 1)[1].strip()
                        break

                if viz_path and os.path.exists(viz_path):
                    # Serve the visualization through our HTTP server
                    viz_url = serve_visualization(viz_path)
                    if viz_url:
                        # Create iframe HTML
                        html_content = f"""
                        <div style="width:100%; height:600px; overflow:hidden; border:1px solid #ddd; border-radius:5px;">
                            <iframe src="{viz_url}" width="100%" height="100%" frameborder="0" allowfullscreen></iframe>
                        </div>
                        <div style="text-align:center; margin-top:10px;">
                            <a href="{viz_url}" target="_blank" style="text-decoration:none; padding:8px 16px;
                               background-color:#f0f0f0; border-radius:4px; color:#333;">
                                Open visualization in new tab
                            </a>
                        </div>
                        """

                        # Add message about visualization
                        history.append(
                            ChatMessage(
                                role="assistant",
                                content="Visualization is ready and displayed below.",
                                metadata={"title": "📊 Visualization Ready"},
                            )
                        )

                        # Return updated history and HTML visualization
                        yield history, gr.update(value=html_content, visible=True)
                    else:
                        history.append(
                            ChatMessage(
                                role="assistant",
                                content=f"Visualization created but couldn't be displayed in UI. You can find it at: {viz_path}",
                                metadata={"title": "📊 Visualization Info"},
                            )
                        )
                        yield history, gr.update(visible=False)
                else:
                    yield history, gr.update(visible=False)
            else:
                yield history, gr.update(visible=False)

        except Exception as e:
            history.append(
                ChatMessage(
                    role="assistant", content=f"Error during analysis: {str(e)}", metadata={"title": "❌ Error"}
                )
            )
            yield history, gr.update(visible=False)

    elif mode == "dataframe":
        if not file_upload:
            history.append(
                ChatMessage(
                    role="assistant",
                    content="Please upload a data file (.csv, .xlsx) for DataFrame mode.",
                    metadata={"title": "❌ Error"},
                )
            )
            yield history, gr.update(visible=False)
            return

        # Notify user
        history.append(
            ChatMessage(
                role="assistant",
                content=f"Processing uploaded file: {file_upload.name}",
                metadata={"title": "🔄 Processing"},
            )
        )
        yield history, gr.update(visible=False)

        try:
            # Load dataframe based on file type
            if file_upload.name.endswith(".csv"):
                df = pd.read_csv(file_upload.name)
            elif file_upload.name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(file_upload.name)
            else:
                history.append(
                    ChatMessage(
                        role="assistant",
                        content="Unsupported file format. Please use .csv, .xls, or .xlsx",
                        metadata={"title": "❌ Error"},
                    )
                )
                yield history, gr.update(visible=False)
                return

            # Stream results
            current_message = ""
            async for message_part in process_user_input_stream(user_input=prompt, data=df, usage_limits=usage_limits):
                current_message += message_part
                history[-1] = ChatMessage(
                    role="assistant", content=current_message, metadata={"title": "🛠️ Analyzing Data"}
                )
                yield history, gr.update(visible=False)
                await asyncio.sleep(0.01)

            # Check if visualization was created
            if "visualization saved" in current_message.lower() or "visualization generated" in current_message.lower():
                viz_path = None
                for line in current_message.split("\n"):
                    if "visualization" in line.lower() and ":" in line:
                        viz_path = line.split(":", 1)[1].strip()
                        break

                if viz_path and os.path.exists(viz_path):
                    # Serve the visualization through our HTTP server
                    viz_url = serve_visualization(viz_path)
                    if viz_url:
                        # Create iframe HTML
                        html_content = f"""
                        <div style="width:100%; height:600px; overflow:hidden; border:1px solid #ddd; border-radius:5px;">
                            <iframe src="{viz_url}" width="100%" height="100%" frameborder="0" allowfullscreen></iframe>
                        </div>
                        <div style="text-align:center; margin-top:10px;">
                            <a href="{viz_url}" target="_blank" style="text-decoration:none; padding:8px 16px;
                               background-color:#f0f0f0; border-radius:4px; color:#333;">
                                Open visualization in new tab
                            </a>
                        </div>
                        """

                        # Add message about visualization
                        history.append(
                            ChatMessage(
                                role="assistant",
                                content="Visualization is ready and displayed below.",
                                metadata={"title": "📊 Visualization Ready"},
                            )
                        )

                        # Return updated history and HTML visualization
                        yield history, gr.update(value=html_content, visible=True)
                    else:
                        history.append(
                            ChatMessage(
                                role="assistant",
                                content=f"Visualization created but couldn't be displayed in UI. You can find it at: {viz_path}",
                                metadata={"title": "📊 Visualization Info"},
                            )
                        )
                        yield history, gr.update(visible=False)
                else:
                    yield history, gr.update(visible=False)
            else:
                yield history, gr.update(visible=False)

        except Exception as e:
            history.append(
                ChatMessage(
                    role="assistant", content=f"Error during analysis: {str(e)}", metadata={"title": "❌ Error"}
                )
            )
            yield history, gr.update(visible=False)
    else:
        history.append(
            ChatMessage(
                role="assistant",
                content="Invalid mode selected. Please choose either 'sql', 'dataframe', or 'auto'.",
                metadata={"title": "❌ Error"},
            )
        )
        yield history, gr.update(visible=False)


def create_gradio_interface():
    """Create and configure the Gradio interface."""
    with gr.Blocks(title="Multi-Agent Data Analysis") as demo:
        # Add custom CSS for visualization container
        gr.HTML("""
        <style>
        .visualization-container {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px 0;
            min-height: 400px;
            overflow: auto;
        }
        #visualization-html iframe {
            width: 100%;
            height: 100%;
            border: none;
        }
        </style>
        """)

        gr.Markdown("# Multi-Agent Data Analysis System")
        gr.Markdown("Upload a data file or provide a database connection string, then ask a question about your data.")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    value=[
                        ChatMessage(
                            role="assistant",
                            content="Welcome to the Multi-Agent Data Analysis System. How can I help you today?",
                        )
                    ],
                    type="messages",
                    height=500,
                    show_copy_button=True,
                )
                # Add HTML component for visualization with better styling
                visualization_html = gr.HTML(
                    visible=False,
                    label="Data Visualization",
                    elem_id="visualization-html",
                    elem_classes="visualization-container",
                )

                with gr.Row():
                    prompt = gr.Textbox(
                        placeholder="Enter your analysis prompt or question...", label="Prompt", scale=8
                    )
                    submit_btn = gr.Button("Submit", scale=1)

            with gr.Column(scale=1):
                mode = gr.Radio(
                    ["auto", "dataframe", "sql"],
                    label="Analysis Mode",
                    info="Choose how to analyze your data",
                    value="auto",
                )
                with gr.Tab("File Upload"):
                    file_upload = gr.File(
                        label="Upload Data File (.csv, .xlsx)", file_types=[".csv", ".xlsx", ".xls"], type="filepath"
                    )

                with gr.Tab("Database"):
                    db_connection = gr.Textbox(
                        placeholder="e.g., sqlite:///data/northwind.db", label="Database Connection String"
                    )

                with gr.Accordion("Advanced Settings", open=False):
                    token_limit = gr.Number(value=4000, label="Token Limit", info="Maximum total token usage")
                    request_limit = gr.Number(value=10, label="Request Limit", info="Maximum number of API requests")

                clear_btn = gr.Button("Clear Chat")
        # Set up event handlers
        submit_btn.click(
            process_query,
            inputs=[chatbot, prompt, mode, file_upload, db_connection, token_limit, request_limit],
            outputs=[chatbot, visualization_html],
        )

        prompt.submit(
            process_query,
            inputs=[chatbot, prompt, mode, file_upload, db_connection, token_limit, request_limit],
            outputs=[chatbot, visualization_html],
        )

        clear_btn.click(lambda: [[], ""], outputs=[chatbot, prompt])

        # Add example prompts
        gr.Examples(
            examples=[
                ["What are the top 5 selling products?"],
                ["Show me sales trends over time"],
                ["Create a pie chart of customer distribution by country"],
                ["Find correlations between columns in this dataset"],
                ["Summarize this data and identify outliers"],
            ],
            inputs=prompt,
        )

    return demo


def main():
    """Main entry point."""  # noqa: D401
    # Check if running as script or as gradio app
    if len(os.sys.argv) > 1:
        # Command-line mode
        args = parse_arguments()
        try:
            result = asyncio.run(run_with_args(args))

            if not args.stream:
                display_results(result)

        except Exception as e:
            print(f"Error: {e}")
    else:  # Gradio UI mode
        demo = create_gradio_interface()
        demo.queue()
        demo.launch(share=False, server_port=8080)  # Use port 8080 instead of the default range


if __name__ == "__main__":
    main()
