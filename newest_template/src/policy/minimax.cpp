#include <utility>
#include "state.hpp"
#include "minimax.hpp"


/*============================================================
 * MiniMax — quiescence
 *
 * Search only capturing moves to stabilize evaluation.
 * Reduces horizon effect.
 *============================================================*/
int MiniMax::quiescence(
    State *state,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p,
    int alpha,
    int beta
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    /* === Lazy move generation === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal checks === */
    if(state->game_state == WIN){
        return P_MAX - ply;
    }
    if(state->game_state == DRAW){
        return 0;
    }

    /* === Standing pat === */
    int stand_pat = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
    
    if(stand_pat >= beta){
        return beta;
    }
    if(stand_pat > alpha){
        alpha = stand_pat;
    }

    /* === Search only capture moves === */
    history.push(state->hash());
    
    int best_score = stand_pat;
    for(auto& action : state->legal_actions){
        /* Check if this is a capture move */
        Point to = action.second;
        int opponent = 1 - state->player;
        if(!state->board.board[opponent][to.first][to.second]){
            /* Not a capture, skip in quiescence */
            continue;
        }

        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();
        
        int score = quiescence(next, history, ply + 1, ctx, p, -beta, -alpha);
        if(!same){
            score = -score;
        }
        
        delete next;

        if(score > best_score){
            best_score = score;
            if(score > alpha){
                alpha = score;
            }
            if(alpha >= beta){
                break; /* Beta cutoff */
            }
        }
    }
    
    history.pop(state->hash());
    return best_score;
}


/*============================================================
 * MiniMax — eval_ctx
 *
 * Negamax with alpha-beta pruning and optional PVS.
 * Caller manages memory.
 *============================================================*/
int MiniMax::eval_ctx(
    State *state,
    int depth,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p,
    int alpha,
    int beta
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    /* === Lazy move generation (sets game_state) === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */
    if(state->game_state == WIN){
        return P_MAX - ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    /* === Repetition check (game-specific) === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }
    history.push(state->hash());

    if(depth <= 0){
        int score;
        if(p.use_quiescence){
            score = quiescence(state, history, ply, ctx, p, alpha, beta);
        } else {
            score = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
        }
        history.pop(state->hash());
        return score;
    }

    /* === Negamax loop with alpha-beta pruning === */
    int best_score = ALPHA_MIN;
    bool first_move = true;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int score;
        if(p.use_pvs && !first_move){
            /* PVS: search non-first moves with narrow window */
            score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -alpha - 1, -alpha);
            if(!same){
                score = -score;
            }
            
            /* If score beats alpha, do full re-search */
            if(score > alpha && score < beta){
                score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -score);
                if(!same){
                    score = -score;
                }
            }
        } else {
            /* Standard alpha-beta search */
            score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
            if(!same){
                score = -score;
            }
        }

        delete next;
        first_move = false;

        if(score > best_score){
            best_score = score;
            if(score > alpha){
                alpha = score;
            }
            if(alpha >= beta){
                break; /* Beta cutoff */
            }
        }
    }

    history.pop(state->hash());
    return best_score;
}

/*============================================================
 * MiniMax — search
 *
 * Iterate legal moves, call eval_ctx with alpha-beta, 
 * return SearchResult.
 *============================================================*/
SearchResult MiniMax::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    int best_score = ALPHA_MIN;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();
    int alpha = ALPHA_MIN;
    int beta = ALPHA_MAX;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();
        
        int score = eval_ctx(next, depth - 1, history, 1, ctx, p, -beta, -alpha);
        if(!same){
            score = -score;
        }

        if(score > best_score){
            best_score = score;
            result.best_move = action;
            if(score > alpha){
                alpha = score;
            }
            if(p.report_partial && ctx.on_root_update){
                ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }

        delete next;
        move_index++;
    }

    result.score = best_score;
    result.nodes = ctx.nodes;
    result.seldepth = ctx.seldepth;
    return result;
} 


/*============================================================
 * MiniMax — default_params / param_defs
 *============================================================*/
ParamMap MiniMax::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
        {"UseAlphaBeta", "true"},
        {"UsePVS", "false"},
        {"UseQuiescence", "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
        {"UseAlphaBeta", ParamDef::CHECK, "true"},
        {"UsePVS", ParamDef::CHECK, "false"},
        {"UseQuiescence", ParamDef::CHECK, "true"},
    };
}
