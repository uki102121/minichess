#include <utility>
#include "state.hpp"
#include "minimax.hpp"
#include <unordered_map>
#include <algorithm>
#include <tuple>


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

    /* === Transposition table lookup === */
    struct TTEntry { int depth; int score; int flag; Move best_move; };
    enum { TT_EXACT=0, TT_LOWER=1, TT_UPPER=2 };
    static std::unordered_map<uint64_t, TTEntry> tt;
    uint64_t key = state->hash();
    auto it = tt.find(key);
    if(it != tt.end()){
        const TTEntry &e = it->second;
        if(e.depth >= depth){
            if(e.flag == TT_EXACT) {
                history.pop(state->hash());
                return e.score;
            } else if(e.flag == TT_LOWER){
                if(e.score > alpha) alpha = e.score;
            } else if(e.flag == TT_UPPER){
                if(e.score < beta) beta = e.score;
            }
            if(alpha >= beta){
                history.pop(state->hash());
                return alpha;
            }
        }
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

    // Move ordering: prefer TT best move and captures
    Move tt_best_move;
    if(it != tt.end()) tt_best_move = it->second.best_move;
    std::vector<std::pair<int, Move>> ordered;
    ordered.reserve(state->legal_actions.size());
    for(auto &action : state->legal_actions){
        int order = 0;
        Point to = action.second;
        int opp = 1 - state->player;
        if(state->board.board[opp][to.first][to.second]) order += 1000; // capture
        if(it != tt.end() && action == tt_best_move) order += 2000; // tt move first
        ordered.push_back({order, action});
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b){ return a.first > b.first; });

    for(auto &pr : ordered){
        auto action = pr.second;
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
                // store TT as lower bound
                TTEntry entry{depth, best_score, TT_LOWER, action};
                tt[key] = entry;
                break; /* Beta cutoff */
            }
        }
    }
    // store TT entry
    int flag = TT_EXACT;
    if(best_score <= alpha) flag = TT_UPPER;
    else if(best_score >= beta) flag = TT_LOWER;
    TTEntry entry{depth, best_score, flag, state->legal_actions.empty() ? Move(Point(0,0),Point(0,0)) : state->legal_actions.empty() ? Move(Point(0,0),Point(0,0)) : state->legal_actions[0]};
    // but prefer stored best_move if we had one
    if(it != tt.end() && it->second.best_move.first != std::pair<size_t,size_t>(0,0)) entry.best_move = it->second.best_move;
    tt[key] = entry;

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